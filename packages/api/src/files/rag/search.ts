import { EModelEndpoint, Tools } from 'librechat-data-provider';
import type { logger } from '@librechat/data-schemas';
import type axios from 'axios';
import type { FileCitationSelectionConfig, selectFileCitationSources } from '../citations';
import type { logAxiosError } from '~/utils/axios';
import { getOutboundTimeoutMs, outboundRequestBudget } from '~/utils/timeouts';

type SearchFile = { file_id: string; fromAgent?: boolean };
type FileSearchResult = [
  string,
  { file_search: { sources: FileSearchSource[]; fileCitations: boolean } } | undefined,
];

type FileSearchSource = {
  type: string;
  fileId: string;
  content: string;
  fileName?: string;
  relevance: number;
  pages: number[];
  pageRelevance: { [page: number]: number };
};

interface SearchHit {
  page_content: string;
  metadata: { source: string; page?: number | string | null; file_id?: string };
}

type SearchResults = [SearchHit, number][];

interface SearchOptions {
  query?: string;
  userId: string;
  files: readonly SearchFile[];
  entity_id?: string;
  fileCitations: boolean;
  appConfig?: {
    endpoints?: Partial<Record<EModelEndpoint, FileCitationSelectionConfig>>;
    timeouts?: { ragMs?: number };
  };
  ragApiUrl?: string;
  httpClient: Pick<typeof axios, 'post'>;
  generateShortLivedToken: (userId: string) => string;
  logAxiosError: typeof logAxiosError;
  selectFileCitationSources: typeof selectFileCitationSources;
  logger: Pick<typeof logger, 'debug'>;
}

/** Chunks one namespace-wide query fetches. Deliberately deeper than the
 * RESULT_LIMIT: selection spends the pool breadth-first across documents and
 * then depth-first within them (see selectHits), and per-document depth past
 * each document's opening chunk only exists in the pool if the query fetched
 * it. The old per-file fan-out (5 per file, merged) implicitly had this depth;
 * a k=10 namespace query would not. */
const NAMESPACE_QUERY_K = 100;

/** How many chunks one tool call returns (upstream contract: the client and
 * the citation instructions both assume the top 10). */
const RESULT_LIMIT = 10;

/** Distinct documents one result set may span before the remaining slots go to
 * more chunks of the documents already selected. Caps sibling-document flooding:
 * corpora hold families of similar documents (one committee's agendas, minutes,
 * reports) whose opening chunks all carry the family's title, so for a title-shaped
 * query every sibling's opening outranks every body chunk and a flat top-10 comes
 * back as one opening per sibling — headers, no content. */
const MAX_FILES = 6;

/** Chunks kept per document while breadth is still available: the opening chunk
 * plus the first body chunk, so each selected document contributes content and
 * not just its header. */
const MAX_PER_FILE = 2;

/** One search hit with the file it came from, ready for ranking. */
type RankedHit = { hit: SearchHit; distance: number; file_id: string };

/** Selects which hits a search returns. Breadth first — up to MAX_PER_FILE chunks
 * from each of the MAX_FILES best-matching documents (documents ordered by their
 * best chunk, chunks within a document by distance) — then a distance-order top-up
 * over the whole pool when breadth and depth together fall short of RESULT_LIMIT.
 * The top-up restores the flat top-10 exactly when it was never the problem: few
 * documents matched, or one document holds all the relevance (its deep chunks fill
 * the remaining slots in distance order). Display order is the selection order, so
 * each document's chunks appear together. */
function selectHits(pool: RankedHit[]): RankedHit[] {
  const byFile = new Map<string, RankedHit[]>();
  for (const entry of pool) {
    const fileHits = byFile.get(entry.file_id);
    if (fileHits) {
      fileHits.push(entry);
    } else {
      byFile.set(entry.file_id, [entry]);
    }
  }
  const fileOrder = [...byFile.values()].sort((a, b) => a[0].distance - b[0].distance);
  const selected: RankedHit[] = [];
  const kept = new Set<RankedHit>();
  for (const fileHits of fileOrder.slice(0, MAX_FILES)) {
    for (const entry of fileHits.slice(0, MAX_PER_FILE)) {
      if (selected.length >= RESULT_LIMIT) break;
      selected.push(entry);
      kept.add(entry);
    }
    if (selected.length >= RESULT_LIMIT) break;
  }
  for (const entry of pool) {
    if (selected.length >= RESULT_LIMIT) break;
    if (kept.has(entry)) {
      continue;
    }
    selected.push(entry);
    kept.add(entry);
  }
  return selected;
}

/** Executes a file-search tool call without sending malformed queries to RAG API. */
export async function executeFileSearchQuery({
  query,
  userId,
  files,
  entity_id,
  fileCitations,
  appConfig,
  ragApiUrl,
  httpClient,
  generateShortLivedToken,
  logAxiosError,
  selectFileCitationSources,
  logger,
}: SearchOptions): Promise<FileSearchResult> {
  if (files.length === 0) {
    return ['No files to search. Instruct the user to add files for the search.', undefined];
  }
  if (typeof query !== 'string' || query.trim().length === 0) {
    return ['A non-empty query is required to search the files.', undefined];
  }
  const jwtToken = generateShortLivedToken(userId);
  if (!jwtToken) {
    return ['There was an error authenticating the file search request.', undefined];
  }

  const headers = {
    Authorization: `Bearer ${jwtToken}`,
    'Content-Type': 'application/json',
  };
  const timeout = getOutboundTimeoutMs(appConfig, 'ragMs');

  // Agent knowledge-base files all live in one namespace under `entity_id`, so one
  // scoped query replaces a request per file. User attachments keep their per-file
  // queries: they carry no shared namespace to search across and must stay user-scoped.
  const namespaceFileIds =
    entity_id == null
      ? []
      : [...new Set(files.filter((file) => file.fromAgent === true).map((file) => file.file_id))];

  /** One entry per issued request: its ranked hits, or null when the request failed. */
  const requests: Array<Promise<RankedHit[] | null>> = [];

  if (namespaceFileIds.length > 0) {
    logger.debug(
      `[${Tools.file_search}] RAG API /query_multiple: ${namespaceFileIds.length} file(s), k=${NAMESPACE_QUERY_K}, entity_id=${entity_id}`,
    );
    requests.push(
      httpClient
        .post<SearchResults>(
          `${ragApiUrl}/query_multiple`,
          { query, file_ids: namespaceFileIds, k: NAMESPACE_QUERY_K, entity_id },
          { headers, ...outboundRequestBudget(timeout, 'ragMs') },
        )
        .then((result) =>
          result.data
            .map(([hit, distance]) => ({ hit, distance, file_id: hit.metadata.file_id }))
            .filter((entry): entry is RankedHit => typeof entry.file_id === 'string'),
        )
        .catch((error) => {
          // /query_multiple answers 404 when the scope matched no chunks; that is an
          // empty result, not a failed search.
          const status = (error as { response?: { status?: number } } | null)?.response?.status;
          if (status === 404) {
            return [];
          }
          logAxiosError({
            message: 'Error encountered in `file_search` while querying the agent knowledge base',
            error,
          });
          return null;
        }),
    );
  }

  for (const file of files) {
    if (entity_id != null && file.fromAgent === true) {
      continue; // already covered by the namespace query
    }
    requests.push(
      httpClient
        .post<SearchResults>(
          `${ragApiUrl}/query`,
          { file_id: file.file_id, query, k: 5 },
          { headers, ...outboundRequestBudget(timeout, 'ragMs') },
        )
        .then((result) =>
          result.data.map(
            ([hit, distance]): RankedHit => ({
              hit,
              distance,
              file_id: file.file_id,
            }),
          ),
        )
        .catch((error) => {
          logAxiosError({
            message: 'Error encountered in `file_search` while querying file',
            error,
          });
          return null;
        }),
    );
  }

  const outcomes = await Promise.all(requests);
  const succeeded = outcomes.filter((outcome): outcome is RankedHit[] => outcome !== null);

  if (succeeded.length === 0) {
    return ['No results found or errors occurred while searching the files.', undefined];
  }

  const formattedResults = selectHits(succeeded.flat().sort((a, b) => a.distance - b.distance)).map(
    (entry) => ({
      filename: entry.hit.metadata.source.split('/').pop(),
      content: entry.hit.page_content,
      distance: entry.distance,
      file_id: entry.file_id,
      page:
        typeof entry.hit.metadata.page === 'number' &&
        Number.isInteger(entry.hit.metadata.page) &&
        entry.hit.metadata.page >= 0
          ? entry.hit.metadata.page + 1
          : null,
    }),
  );

  if (formattedResults.length === 0) {
    return [
      'No content found in the files. The files may not have been processed correctly or you may need to refine your query.',
      undefined,
    ];
  }

  const sources = formattedResults.map((result) => ({
    type: 'file',
    fileId: result.file_id,
    content: result.content,
    fileName: result.filename,
    relevance: 1.0 - result.distance,
    pages: result.page ? [result.page] : [],
    pageRelevance: result.page ? { [result.page]: 1.0 - result.distance } : {},
  }));

  const citationConfig = appConfig?.endpoints?.[EModelEndpoint.agents];
  const citationSources = fileCitations ? selectFileCitationSources(sources, citationConfig) : [];
  const formattedString = formattedResults
    .map((result, index) => {
      const citationIndex = citationSources.indexOf(sources[index]);
      return `File: ${result.filename}${
        citationIndex >= 0 ? `\nAnchor: \\ue202turn0file${citationIndex} (${result.filename})` : ''
      }\nRelevance: ${(1.0 - result.distance).toFixed(4)}\nContent: ${result.content}\n`;
    })
    .join('\n---\n');

  return [formattedString, { [Tools.file_search]: { sources, fileCitations } }];
}
