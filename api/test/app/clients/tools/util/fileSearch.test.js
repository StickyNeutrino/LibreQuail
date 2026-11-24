const axios = require('axios');
const { ResourceType } = require('librechat-data-provider');

jest.mock('axios');
jest.mock('@librechat/api', () => {
  const { selectFileCitationSources, executeFileSearchQuery } =
    jest.requireActual('@librechat/api');
  return {
    generateShortLivedToken: jest.fn(),
    logAxiosError: jest.fn(),
    selectFileCitationSources,
    executeFileSearchQuery,
  };
});

jest.mock('@librechat/data-schemas', () => ({
  logger: {
    warn: jest.fn(),
    error: jest.fn(),
    debug: jest.fn(),
  },
}));

jest.mock('~/models', () => ({
  getFiles: jest.fn().mockResolvedValue([]),
}));

jest.mock('~/server/services/Files/permissions', () => ({
  filterFilesByAgentAccess: jest.fn((options) => Promise.resolve(options.files)),
}));

const { createFileSearchTool, primeFiles } = require('~/app/clients/tools/util/fileSearch');
const { generateShortLivedToken } = require('@librechat/api');

describe('fileSearch.js - agent file authorization', () => {
  it('uses the permission resource type established by the calling route', async () => {
    const { getFiles } = require('~/models');
    const { filterFilesByAgentAccess } = require('~/server/services/Files/permissions');
    const files = [{ file_id: 'owner-file', filename: 'owner.pdf', user: 'agent-owner' }];
    getFiles.mockResolvedValueOnce(files);

    await primeFiles({
      req: { user: { id: 'remote-viewer', role: 'USER' } },
      agentId: 'agent-123',
      agentResourceType: ResourceType.REMOTE_AGENT,
      tool_resources: { file_search: { file_ids: ['owner-file'] } },
    });

    expect(filterFilesByAgentAccess).toHaveBeenCalledWith({
      files,
      userId: 'remote-viewer',
      role: 'USER',
      agentId: 'agent-123',
      resourceType: ResourceType.REMOTE_AGENT,
    });
  });
});

describe('fileSearch.js - tuple return validation', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    process.env.RAG_API_URL = 'http://localhost:8000';
  });

  describe('error cases should return tuple with undefined as second value', () => {
    it('should return tuple when no files provided', async () => {
      const fileSearchTool = await createFileSearchTool({
        userId: 'user1',
        files: [],
      });

      const result = await fileSearchTool.func({ query: 'test query' });

      expect(Array.isArray(result)).toBe(true);
      expect(result).toHaveLength(2);
      expect(result[0]).toBe('No files to search. Instruct the user to add files for the search.');
      expect(result[1]).toBeUndefined();
    });

    it.each(['', '   ', '\n\t', undefined])(
      'does not query any files when the search query is blank (%j)',
      async (query) => {
        const fileSearchTool = await createFileSearchTool({
          userId: 'user1',
          files: [
            { file_id: 'file-1', filename: 'one.pdf' },
            { file_id: 'file-2', filename: 'two.pdf' },
          ],
        });

        const result = await fileSearchTool.func({ query });

        expect(result).toEqual(['A non-empty query is required to search the files.', undefined]);
        expect(axios.post).not.toHaveBeenCalled();
        expect(generateShortLivedToken).not.toHaveBeenCalled();
      },
    );

    it('should return tuple when JWT token generation fails', async () => {
      generateShortLivedToken.mockReturnValue(null);

      const fileSearchTool = await createFileSearchTool({
        userId: 'user1',
        files: [{ file_id: 'file-1', filename: 'test.pdf' }],
      });

      const result = await fileSearchTool.func({ query: 'test query' });

      expect(Array.isArray(result)).toBe(true);
      expect(result).toHaveLength(2);
      expect(result[0]).toBe('There was an error authenticating the file search request.');
      expect(result[1]).toBeUndefined();
    });

    it('should return tuple when no valid results found', async () => {
      generateShortLivedToken.mockReturnValue('mock-jwt-token');
      axios.post.mockRejectedValue(new Error('API Error'));

      const fileSearchTool = await createFileSearchTool({
        userId: 'user1',
        files: [{ file_id: 'file-1', filename: 'test.pdf' }],
      });

      const result = await fileSearchTool.func({ query: 'test query' });

      expect(Array.isArray(result)).toBe(true);
      expect(result).toHaveLength(2);
      expect(result[0]).toBe('No results found or errors occurred while searching the files.');
      expect(result[1]).toBeUndefined();
    });
  });

  describe('success cases should return tuple with artifact object', () => {
    it.each([
      [0, [1]],
      [2, [3]],
      [undefined, []],
      [null, []],
      [-1, []],
      [1.5, []],
      ['2', []],
    ])('maps RAG page index %s to citation pages %j', async (page, pages) => {
      generateShortLivedToken.mockReturnValue('mock-jwt-token');
      axios.post.mockResolvedValue({
        data: [
          [{ page_content: 'Synthetic passage', metadata: { source: '/test.pdf', page } }, 0.2],
        ],
      });

      const fileSearchTool = await createFileSearchTool({
        userId: 'user1',
        files: [{ file_id: 'file-123', filename: 'test.pdf' }],
        fileCitations: true,
      });
      const [, artifact] = await fileSearchTool.func({ query: 'Synthetic page check' });
      const source = artifact.file_search.sources[0];

      expect(source.pages).toEqual(pages);
      expect(source.pageRelevance).toEqual(pages.length ? { [pages[0]]: expect.any(Number) } : {});
    });

    it('forwards valid query text unchanged', async () => {
      generateShortLivedToken.mockReturnValue('mock-jwt-token');
      axios.post.mockResolvedValue({ data: [] });
      const fileSearchTool = await createFileSearchTool({
        userId: 'user1',
        files: [{ file_id: 'file-1', filename: 'one.pdf' }],
      });

      await fileSearchTool.func({ query: '  find me  ' });

      expect(axios.post).toHaveBeenCalledWith(
        'http://localhost:8000/query',
        { file_id: 'file-1', query: '  find me  ', k: 5 },
        {
          headers: {
            Authorization: 'Bearer mock-jwt-token',
            'Content-Type': 'application/json',
          },
          timeout: 30_000,
          signal: expect.any(AbortSignal),
          timeoutErrorMessage: expect.any(String),
        },
      );
    });

    it('should return tuple with formatted results and sources artifact', async () => {
      generateShortLivedToken.mockReturnValue('mock-jwt-token');

      const mockApiResponse = {
        data: [
          [
            {
              page_content: 'This is test content from the document',
              metadata: { source: '/path/to/test.pdf', page: 0 },
            },
            0.2,
          ],
          [
            {
              page_content: 'Additional relevant content',
              metadata: { source: '/path/to/test.pdf', page: 1 },
            },
            0.35,
          ],
        ],
      };

      axios.post.mockResolvedValue(mockApiResponse);

      const fileSearchTool = await createFileSearchTool({
        userId: 'user1',
        files: [{ file_id: 'file-123', filename: 'test.pdf' }],
        entity_id: 'agent-456',
      });

      const result = await fileSearchTool.func({ query: 'test query' });

      expect(Array.isArray(result)).toBe(true);
      expect(result).toHaveLength(2);

      const [formattedString, artifact] = result;

      expect(typeof formattedString).toBe('string');
      expect(formattedString).toContain('File: test.pdf');
      expect(formattedString).toContain('Relevance:');
      expect(formattedString).toContain('This is test content from the document');
      expect(formattedString).toContain('Additional relevant content');

      expect(artifact).toBeDefined();
      expect(artifact).toHaveProperty('file_search');
      expect(artifact.file_search).toHaveProperty('sources');
      expect(artifact.file_search).toHaveProperty('fileCitations', false);
      expect(Array.isArray(artifact.file_search.sources)).toBe(true);
      expect(artifact.file_search.sources.length).toBe(2);

      const source = artifact.file_search.sources[0];
      expect(source).toMatchObject({
        type: 'file',
        fileId: 'file-123',
        fileName: 'test.pdf',
        content: expect.any(String),
        relevance: expect.any(Number),
        pages: [1],
        pageRelevance: { 1: expect.any(Number) },
      });
    });

    it('should include file citations in description when enabled', async () => {
      generateShortLivedToken.mockReturnValue('mock-jwt-token');

      const mockApiResponse = {
        data: [
          [
            {
              page_content: 'Content with citations',
              metadata: { source: '/path/to/doc.pdf', page: 3 },
            },
            0.15,
          ],
        ],
      };

      axios.post.mockResolvedValue(mockApiResponse);

      const fileSearchTool = await createFileSearchTool({
        userId: 'user1',
        files: [{ file_id: 'file-789', filename: 'doc.pdf' }],
        fileCitations: true,
      });

      const result = await fileSearchTool.func({ query: 'test query' });

      expect(Array.isArray(result)).toBe(true);
      expect(result).toHaveLength(2);

      const [formattedString, artifact] = result;

      expect(formattedString).toContain('Anchor:');
      expect(formattedString).toContain('\\ue202turn0file0');
      expect(artifact.file_search.fileCitations).toBe(true);
    });

    it('keeps a successful file result when another file search fails', async () => {
      generateShortLivedToken.mockReturnValue('mock-jwt-token');
      axios.post.mockRejectedValueOnce(new Error('file unavailable')).mockResolvedValueOnce({
        data: [[{ page_content: 'Found passage', metadata: { source: '/good.pdf' } }, 0.2]],
      });
      const fileSearchTool = await createFileSearchTool({
        userId: 'user1',
        files: [
          { file_id: 'missing', filename: 'missing.pdf' },
          { file_id: 'good', filename: 'good.pdf' },
        ],
      });

      const [content, artifact] = await fileSearchTool.func({ query: 'lookup' });

      expect(axios.post).toHaveBeenCalledTimes(2);
      expect(content).toContain('Found passage');
      expect(artifact.file_search.sources).toHaveLength(1);
      expect(artifact.file_search.sources[0].fileId).toBe('good');
    });

    it('should handle multiple files correctly', async () => {
      generateShortLivedToken.mockReturnValue('mock-jwt-token');

      const mockResponse1 = {
        data: [
          [
            {
              page_content: 'Content from file 1',
              metadata: { source: '/path/to/file1.pdf', page: 1 },
            },
            0.25,
          ],
        ],
      };

      const mockResponse2 = {
        data: [
          [
            {
              page_content: 'Content from file 2',
              metadata: { source: '/path/to/file2.pdf', page: 1 },
            },
            0.15,
          ],
        ],
      };

      axios.post.mockResolvedValueOnce(mockResponse1).mockResolvedValueOnce(mockResponse2);

      const fileSearchTool = await createFileSearchTool({
        userId: 'user1',
        files: [
          { file_id: 'file-1', filename: 'file1.pdf' },
          { file_id: 'file-2', filename: 'file2.pdf' },
        ],
      });

      const result = await fileSearchTool.func({ query: 'test query' });

      expect(Array.isArray(result)).toBe(true);
      expect(result).toHaveLength(2);

      const [formattedString, artifact] = result;

      expect(formattedString).toContain('file1.pdf');
      expect(formattedString).toContain('file2.pdf');
      expect(artifact.file_search.sources).toHaveLength(2);
      // Results are sorted by distance (ascending), so file-2 (0.15) comes before file-1 (0.25)
      expect(artifact.file_search.sources[0].fileId).toBe('file-2');
      expect(artifact.file_search.sources[1].fileId).toBe('file-1');
    });
  });
});

describe('entity_id scoping by file origin', () => {
  const ORIGINAL_RAG_API_URL = process.env.RAG_API_URL;

  beforeEach(() => {
    jest.clearAllMocks();
    process.env.RAG_API_URL = 'http://localhost:8000';
    generateShortLivedToken.mockReturnValue('mock-jwt-token');
    axios.post.mockResolvedValue({ data: [] });
  });

  afterEach(() => {
    if (ORIGINAL_RAG_API_URL === undefined) {
      delete process.env.RAG_API_URL;
    } else {
      process.env.RAG_API_URL = ORIGINAL_RAG_API_URL;
    }
  });

  function bodiesSent() {
    return axios.post.mock.calls
      .filter(([url]) => String(url).endsWith('/query'))
      .map(([, body]) => body);
  }

  function batchBodiesSent() {
    return axios.post.mock.calls
      .filter(([url]) => String(url).endsWith('/query_multiple'))
      .map(([, body]) => body);
  }

  it('searches agent knowledge-base files as one namespace query', async () => {
    const tool = await createFileSearchTool({
      userId: 'user1',
      entity_id: 'agent_123',
      files: [
        { file_id: 'kb-1', filename: 'kb.pdf', fromAgent: true },
        { file_id: 'kb-2', filename: 'kb2.pdf', fromAgent: true },
        { file_id: 'user-1', filename: 'attachment.txt', fromAgent: false },
      ],
    });
    await tool.func({ query: 'q' });

    // The agent's files share one namespace, so one scoped request covers them all;
    // the user attachment is not part of that namespace and keeps its own request.
    // k mirrors NAMESPACE_QUERY_K — deeper than the 10-result limit so selection
    // can spend the pool breadth-first across documents, then depth-first within them.
    expect(batchBodiesSent()).toEqual([
      { query: 'q', file_ids: ['kb-1', 'kb-2'], k: 100, entity_id: 'agent_123' },
    ]);
    expect(bodiesSent()).toEqual([{ file_id: 'user-1', query: 'q', k: 5 }]);
  });

  it('keeps the per-file fan-out when there is no namespace', async () => {
    const tool = await createFileSearchTool({
      userId: 'user1',
      files: [
        { file_id: 'kb-1', filename: 'kb.pdf', fromAgent: true },
        { file_id: 'kb-2', filename: 'kb2.pdf', fromAgent: true },
      ],
    });
    await tool.func({ query: 'q' });
    expect(batchBodiesSent()).toEqual([]);
    expect(bodiesSent()).toHaveLength(2);
  });

  it('omits entity_id when fromAgent is not set (safe default)', async () => {
    const tool = await createFileSearchTool({
      userId: 'user1',
      entity_id: 'agent_123',
      files: [{ file_id: 'legacy-1', filename: 'legacy.pdf' }],
    });
    await tool.func({ query: 'q' });
    expect(batchBodiesSent()).toEqual([]);
    expect(bodiesSent()[0].entity_id).toBeUndefined();
  });

  it('sends no entity_id when none is provided', async () => {
    const tool = await createFileSearchTool({
      userId: 'user1',
      files: [{ file_id: 'f1', filename: 'a.txt', fromAgent: true }],
    });
    await tool.func({ query: 'q' });
    expect(batchBodiesSent()).toEqual([]);
    expect(bodiesSent()[0].entity_id).toBeUndefined();
  });

  it('takes each namespace hit fileId from the chunk metadata', async () => {
    axios.post.mockResolvedValue({
      data: [
        [{ page_content: 'closer match', metadata: { source: '/kb2.pdf', file_id: 'kb-2' } }, 0.1],
        [{ page_content: 'farther match', metadata: { source: '/kb1.pdf', file_id: 'kb-1' } }, 0.2],
      ],
    });
    const tool = await createFileSearchTool({
      userId: 'user1',
      entity_id: 'agent_123',
      files: [{ file_id: 'kb-1', filename: 'kb1.pdf', fromAgent: true }],
    });

    const [, artifact] = await tool.func({ query: 'q' });

    expect(artifact.file_search.sources[0].fileId).toBe('kb-2');
    expect(artifact.file_search.sources[1].fileId).toBe('kb-1');
  });

  it('reads a namespace 404 as no matches rather than a failed search', async () => {
    axios.post.mockRejectedValue({ response: { status: 404 } });
    const tool = await createFileSearchTool({
      userId: 'user1',
      entity_id: 'agent_123',
      files: [{ file_id: 'kb-1', filename: 'kb.pdf', fromAgent: true }],
    });

    const [content] = await tool.func({ query: 'q' });

    expect(content).toBe(
      'No content found in the files. The files may not have been processed correctly or you may need to refine your query.',
    );
  });

  it('keeps user-attachment results when the namespace query fails', async () => {
    axios.post.mockImplementation(async (url) => {
      if (String(url).endsWith('/query_multiple')) {
        throw new Error('namespace unavailable');
      }
      return {
        data: [[{ page_content: 'attachment passage', metadata: { source: '/att.txt' } }, 0.2]],
      };
    });
    const tool = await createFileSearchTool({
      userId: 'user1',
      entity_id: 'agent_123',
      files: [
        { file_id: 'kb-1', filename: 'kb.pdf', fromAgent: true },
        { file_id: 'user-1', filename: 'att.txt', fromAgent: false },
      ],
    });

    const [content, artifact] = await tool.func({ query: 'q' });

    expect(content).toContain('attachment passage');
    expect(artifact.file_search.sources).toHaveLength(1);
    expect(artifact.file_search.sources[0].fileId).toBe('user-1');
  });
});

describe('fileSearch.js - result selection (breadth + depth)', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    process.env.RAG_API_URL = 'http://localhost:8000';
    generateShortLivedToken.mockReturnValue('mock-jwt-token');
  });

  const kbFile = (id) => ({ file_id: id, filename: `${id}.txt`, fromAgent: true });

  it('caps sibling-document openings so body chunks make the top 10', async () => {
    // A corpus of sibling documents (one committee's agendas and minutes) all
    // carry the family title in their opening chunk: a flat top-10 would return
    // ten openings and no content, which is how the 2026-11-24 "Tecolote Canyon
    // Committee Overview" chat ended up answering from web search.
    const data = [];
    const files = [];
    for (let f = 0; f < 10; f++) {
      files.push(kbFile(`doc-${f}`));
      data.push([
        {
          page_content: `doc ${f} opening`,
          metadata: { source: `/doc-${f}.txt`, file_id: `doc-${f}` },
        },
        0.01 + f * 0.001,
      ]);
      for (let c = 1; c <= 4; c++) {
        data.push([
          {
            page_content: `doc ${f} body chunk ${c}`,
            metadata: { source: `/doc-${f}.txt`, file_id: `doc-${f}` },
          },
          0.1 + f * 0.001 + c * 0.01,
        ]);
      }
    }
    axios.post.mockResolvedValue({ data });

    const tool = await createFileSearchTool({
      userId: 'user1',
      entity_id: 'agent_123',
      files,
    });
    const [content, artifact] = await tool.func({ query: 'the committee' });

    const sources = artifact.file_search.sources;
    expect(sources).toHaveLength(10);
    // At most MAX_FILES documents, and no document contributes more than its
    // opening plus its first body chunk — deeper chunks wait for a follow-up
    // search or a full-document read instead of being crowded out entirely.
    const counts = {};
    for (const source of sources) {
      counts[source.fileId] = (counts[source.fileId] ?? 0) + 1;
    }
    expect(Object.keys(counts).length).toBeLessThanOrEqual(6);
    expect(Object.values(counts).every((n) => n <= 2)).toBe(true);
    // Body content is in the result set, not just openings.
    expect(content).toContain('doc 0 body chunk 1');
    expect(content).toContain('doc 4 body chunk 1');
    expect(content).not.toContain('body chunk 2');
    // Display order groups each document's chunks together (opening first).
    expect(sources[0].content).toBe('doc 0 opening');
    expect(sources[1].content).toBe('doc 0 body chunk 1');
    expect(sources[2].content).toBe('doc 1 opening');
  });

  it('keeps the flat top-10 when one document holds all the relevance', async () => {
    const data = [];
    for (let c = 0; c < 12; c++) {
      data.push([
        { page_content: `chunk ${c}`, metadata: { source: '/single.txt', file_id: 'single' } },
        0.1 + c * 0.01,
      ]);
    }
    axios.post.mockResolvedValue({ data });

    const tool = await createFileSearchTool({
      userId: 'user1',
      entity_id: 'agent_123',
      files: [kbFile('single')],
    });
    const [content, artifact] = await tool.func({ query: 'specific detail' });

    expect(artifact.file_search.sources).toHaveLength(10);
    expect(content).toContain('chunk 9');
    expect(content).not.toContain('chunk 10');
    // Distance order is preserved: the top-up fills the slots the breadth pass
    // left open with the pool's next-best chunks.
    const contentOrder = artifact.file_search.sources.map((s) => s.content);
    expect(contentOrder).toEqual(Array.from({ length: 10 }, (_, c) => `chunk ${c}`));
  });

  it('returns every hit when the pool is smaller than the result limit', async () => {
    const data = [];
    for (const fileId of ['a', 'b']) {
      for (let c = 0; c < 3; c++) {
        data.push([
          {
            page_content: `${fileId} chunk ${c}`,
            metadata: { source: `/${fileId}.txt`, file_id: fileId },
          },
          0.1 + (fileId === 'b' ? 0.05 : 0) + c * 0.01,
        ]);
      }
    }
    axios.post.mockResolvedValue({ data });

    const tool = await createFileSearchTool({
      userId: 'user1',
      entity_id: 'agent_123',
      files: [kbFile('a'), kbFile('b')],
    });
    const [content, artifact] = await tool.func({ query: 'q' });

    expect(artifact.file_search.sources).toHaveLength(6);
    expect(content).toContain('a chunk 2');
    expect(content).toContain('b chunk 2');
  });
});

describe('fileSearch.js - citation anchor ordinals', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    process.env.RAG_API_URL = 'http://localhost:8000';
    generateShortLivedToken.mockReturnValue('mock-jwt-token');
  });

  const mockQueryResponse = () => {
    axios.post.mockImplementation(async (_url, body) => ({
      data: [
        [
          {
            page_content: body.file_id === 'file-1' ? 'alpha file passage' : 'beta file passage',
            metadata: {
              source: body.file_id === 'file-1' ? 'uploads/alpha.pdf' : 'uploads/beta.pdf',
            },
          },
          0.1,
        ],
      ],
    }));
  };

  const twoFiles = [
    { file_id: 'file-1', filename: 'alpha.pdf', fromAgent: true },
    { file_id: 'file-2', filename: 'beta.pdf', fromAgent: true },
  ];

  it('numbers anchors by result-set ordinal across multiple calls in one turn', async () => {
    mockQueryResponse();
    const fileSearchTool = await createFileSearchTool({
      userId: 'user1',
      fileCitations: true,
      files: twoFiles,
    });

    const [firstContent, firstArtifact] = await fileSearchTool.func({ query: 'first query' });
    expect(firstArtifact.file_search.sources).toHaveLength(2);
    expect(firstContent).toContain('\\ue202turn0file0');
    expect(firstContent).toContain('\\ue202turn0file1');
    expect(firstContent).not.toContain('\\ue202turn1file');

    const [secondContent, secondArtifact] = await fileSearchTool.func({ query: 'second query' });
    expect(secondArtifact.file_search.sources).toHaveLength(2);
    expect(secondContent).toContain('\\ue202turn1file0');
    expect(secondContent).toContain('\\ue202turn1file1');
    expect(secondContent).not.toContain('\\ue202turn0file');
  });

  it('does not advance the ordinal for a call that produces no sources', async () => {
    mockQueryResponse();
    // Each file_search call fires one POST per file — fail all of them so the
    // tool call produces no sources (and no citation attachment).
    axios.post.mockRejectedValueOnce(new Error('API Error'));
    axios.post.mockRejectedValueOnce(new Error('API Error'));
    const fileSearchTool = await createFileSearchTool({
      userId: 'user1',
      fileCitations: true,
      files: twoFiles,
    });

    const [failedContent, failedArtifact] = await fileSearchTool.func({ query: 'failing query' });
    expect(failedArtifact).toBeUndefined();
    expect(failedContent).toBe('No results found or errors occurred while searching the files.');

    const [nextContent] = await fileSearchTool.func({ query: 'next query' });
    expect(nextContent).toContain('\\ue202turn0file0');
    expect(nextContent).not.toContain('\\ue202turn1file');
  });
});
