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

/** How many chunks one namespace-wide query returns. The tool call keeps the top 10
 * chunks overall, and a single scoped query returns that set directly: the old
 * per-file fan-out (5 per file, merged) picked the same chunks at one request
 * per knowledge-base file. */
const NAMESPACE_QUERY_K = 10;

/** One search hit with the file it came from, ready for ranking. */
type RankedHit = { hit: SearchHit; distance: number; file_id: string };

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

  const formattedResults = succeeded
    .flat()
    .sort((a, b) => a.distance - b.distance)
    .slice(0, 10)
    .map((entry) => ({
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
    }));

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
