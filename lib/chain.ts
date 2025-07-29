import { config } from 'dotenv';
import { ChatTogetherAI } from '@langchain/community/chat_models/togetherai';
import { RunnableSequence } from '@langchain/core/runnables';
import { PromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { Pool, PoolClient } from 'pg';
import { QdrantClient } from '@qdrant/js-client-rest';
import { QdrantVectorStore } from '@langchain/community/vectorstores/qdrant';
import { TogetherAIEmbeddings } from '@langchain/community/embeddings/togetherai';

// Load environment variables
config();
const logger = {
  info: (...args: any[]) => console.log('[INFO]', ...args),
  error: (...args: any[]) => console.error('[ERROR]', ...args),
  warn: (...args: any[]) => console.warn('[WARN]', ...args),
};

// Validate environment variables
function validateEnvVars() {
  const requiredVars = [
    'POSTGRES_URL',
    'QDRANT_URL',
    'QDRANT_API_KEY',
    'TOGETHERAI_API_KEY',
  ];
  const missingVars = requiredVars.filter(varName => !process.env[varName]);
  if (missingVars.length > 0) {
    throw new Error(`Missing required environment variables: ${missingVars.join(', ')}`);
  }
}
validateEnvVars();

// Interfaces for type safety
interface Entities {
  teams: string[];
  players: string[];
  years: string[];
  venues: string[];
  matchTypes: string[];
  competitions: string[];
  match_id?: string;
}

interface Context {
  source: string;
  title: string;
  data: any[];
}

interface QueryAnalysis {
  type: string;
  entities: Entities;
  searchStrategy: string[];
  confidence: number;
  needsVector: boolean;
  needsSQL: boolean;
  priority: string;
}

// Database connection with connection pooling
const postgres = new Pool({
  connectionString: process.env.POSTGRES_URL,
  ssl: { rejectUnauthorized: false },
  max: 10,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 10000,
  keepAlive: true,
  keepAliveInitialDelayMillis: 0,
});

// Together AI Mistral LLM
const llm = new ChatTogetherAI({
  model: 'mistralai/Mistral-7B-Instruct-v0.2',
  apiKey: process.env.TOGETHERAI_API_KEY,
  temperature: 0.1, // Lowered for factual accuracy
  maxTokens: 1200,
  topP: 0.9,
  streaming: false,
});

// Enhanced embeddings
const embeddings = new TogetherAIEmbeddings({
  model: 'BAAI/bge-large-en-v1.5',
  apiKey: process.env.TOGETHERAI_API_KEY,
  batchSize: 512,
});

// Qdrant client
const qdrant = new QdrantClient({
  url: process.env.QDRANT_URL!,
  apiKey: process.env.QDRANT_API_KEY!,
});

// Initialize vector store
let vectorStore: QdrantVectorStore | null = null;

async function getVectorStore() {
  if (!vectorStore) {
    try {
      logger.info('Initializing Qdrant vector store...');
      const collections = await Promise.race([
        qdrant.getCollections(),
        new Promise((_, reject) => setTimeout(() => reject(new Error('Qdrant connection timeout')), 5000)),
      ]) as { collections: { name: string }[] };
      const collectionExists = collections.collections.some(
        (col: any) => col.name === 'match_summaries',
      );
      if (!collectionExists) {
        await qdrant.createCollection('match_summaries', {
          vectors: { size: 1536, distance: 'Cosine' },
        });
        logger.info('✅ Created Qdrant collection: match_summaries');
      }
      vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
        collectionName: 'match_summaries',
        client: qdrant,
      });
      logger.info('✅ Vector store initialized successfully');
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown vector error';
      logger.error(`❌ Vector store initialization failed: ${errorMessage}`);
      throw new Error(`Failed to initialize vector store: ${errorMessage}`);
    }
  }
  return vectorStore;
}

// Query analysis with match ID detection
export async function analyzeQuery(question: string): Promise<QueryAnalysis> {
  const lowerQuestion = question.toLowerCase();
  const statKeywords = ['average', 'strike rate', 'economy', 'wicket', 'run', 'score', 'record', 'stat', 'performance', 'total', 'highest', 'best', 'career', 'statistics'];
  const matchKeywords = ['match', 'game', 'final', 'semifinal', 'qualifier', 'eliminator', 'tournament', 'series', 'encounter', 'result', 'outcome'];
  const teamKeywords = ['team', 'franchise', 'squad', 'club', 'karachi', 'lahore', 'islamabad', 'peshawar', 'quetta', 'multan', 'kings', 'qalandars', 'united', 'zalmi', 'gladiators', 'sultans'];
  const comparisonKeywords = ['vs', 'versus', 'against', 'compare', 'better', 'difference', 'head to head', 'h2h', 'rivalry'];
  const venueKeywords = ['venue', 'stadium', 'ground', 'national stadium', 'gaddafi', 'rawalpindi', 'karachi', 'lahore'];

  let type = 'general';
  let confidence = 0;
  const entities = await extractEntities(question);

  // Match ID detection
  const matchIdMatch = question.match(/match\s+(\d+)/i);
  if (matchIdMatch) {
    type = 'match_report';
    entities.match_id = matchIdMatch[1];
    confidence = 1.0;
  } else {
    const typeChecks = [
      { keywords: statKeywords, type: 'stats' },
      { keywords: matchKeywords, type: 'match_report' },
      { keywords: teamKeywords, type: 'team_info' },
      { keywords: comparisonKeywords, type: 'comparison' },
      { keywords: venueKeywords, type: 'venue_info' },
    ];
    typeChecks.forEach(check => {
      const matches = check.keywords.filter(keyword => lowerQuestion.includes(keyword)).length;
      const score = matches / check.keywords.length;
      if (score > confidence) {
        confidence = score;
        type = check.type;
      }
    });
  }

  const searchStrategy = determineSearchStrategy(type, entities, question);
  logger.info(`Query analysis: type=${type}, entities=${JSON.stringify(entities)}`);
  return {
    type,
    entities,
    searchStrategy,
    confidence,
    needsVector: type === 'general' && !entities.match_id,
    needsSQL: type !== 'general' || Boolean(entities.match_id),
    priority: type === 'stats' || type === 'match_report' || entities.match_id ? 'sql' : 'vector',
  };
}

// Dynamic entity extraction with exact player matching
async function extractEntities(question: string): Promise<Entities> {
  const entities: Entities = {
    teams: [], players: [], years: [], venues: [], matchTypes: [], competitions: [], match_id: undefined,
  };
  const lowerQuestion = question.toLowerCase();
  let client: PoolClient | undefined;

  try {
    client = await postgres.connect();

    // Team extraction
    const teamQuery = `
      SELECT DISTINCT team_name
      FROM (
        SELECT team1 AS team_name FROM matches
        UNION
        SELECT team2 AS team_name FROM matches
      ) teams
      WHERE team_name IS NOT NULL
    `;
    const teamResult = await client.query(teamQuery);
    const teams = teamResult.rows.map(row => ({
      patterns: [
        row.team_name.toLowerCase(),
        ...row.team_name.toLowerCase().split(' ').filter((w: string) => w.length > 2),
      ],
      name: row.team_name,
    }));
    teams.forEach(team => {
      if (team.patterns.some(pattern => lowerQuestion.includes(pattern))) {
        entities.teams.push(team.name);
      }
    });

    // Exact player extraction
    const playerQuery = `
      SELECT DISTINCT player_name
      FROM player_stats
      WHERE player_name IS NOT NULL
    `;
    const playerResult = await client.query(playerQuery);
    const players = playerResult.rows.map(row => row.player_name);
    players.forEach((player: string) => {
      if (lowerQuestion.includes(player.toLowerCase())) {
        entities.players.push(player);
      }
    });

    // Year and season detection
    const yearMatches = question.match(/\b(20\d{2})\b/g);
    if (yearMatches) {
      entities.years = [...new Set(yearMatches)];
    }
    const editionMatch = lowerQuestion.match(/psl\s*(\d+)/i);
    if (editionMatch) {
      const edition = parseInt(editionMatch[1]);
      const year = 2015 + edition;
      if (year >= 2016 && year <= 2024) {
        entities.years.push(year.toString());
      }
    }

    // Match ID
    const matchIdMatch = question.match(/match\s+(\d+)/i);
    if (matchIdMatch) {
      entities.match_id = matchIdMatch[1];
    }

    // Match types
    const matchTypes = ['final', 'grand final', 'championship', 'semifinal', 'semi final', 'semi-final', 'qualifier', 'eliminator', 'playoff', 'playoffs'];
    matchTypes.forEach(type => {
      if (lowerQuestion.includes(type)) {
        entities.matchTypes.push(type);
      }
    });

    // Venues
    const venueQuery = `
      SELECT DISTINCT venue
      FROM matches
      WHERE venue IS NOT NULL
    `;
    const venueResult = await client.query(venueQuery);
    const venues = venueResult.rows.map(row => row.venue.toLowerCase());
    venues.forEach((venue: string) => {
      if (lowerQuestion.includes(venue)) {
        entities.venues.push(venue);
      }
    });

  } catch (error: unknown) {
    logger.error(`❌ Entity extraction error: ${error instanceof Error ? error.message : 'Unknown error'}`);
  } finally {
    if (client) client.release();
  }

  return entities;
}

// Search strategy determination
function determineSearchStrategy(type: string, entities: Entities, question: string): string[] {
  const strategies = [];
  const lowerQuestion = question.toLowerCase();

  if (entities.match_id) {
    strategies.push('sql_match_id');
  }
  if (entities.teams.length > 0 || entities.players.length > 0 || entities.years.length > 0) {
    strategies.push('sql_specific');
  }
  if (type === 'stats' || lowerQuestion.includes('statistics') || lowerQuestion.includes('performance')) {
    strategies.push('sql_stats');
  }
  if (lowerQuestion.includes('recent') || lowerQuestion.includes('latest') || lowerQuestion.includes('current')) {
    strategies.push('sql_recent');
  }
  if (entities.matchTypes.length > 0 || lowerQuestion.includes('championship') || lowerQuestion.includes('winner')) {
    strategies.push('sql_championship');
  }
  if (entities.venues.length > 0 || type === 'venue_info') {
    strategies.push('sql_venue');
  }
  if (entities.competitions.some(c => c.startsWith('score:'))) {
    strategies.push('sql_score');
  }
  if (type === 'general' && !entities.match_id) {
    strategies.push('vector');
  }

  return strategies;
}

// Enhanced context retrieval with match ID priority
async function getEnhancedContext(input: { question: string; queryType?: string; entities?: Entities }): Promise<string> {
  const { question, queryType = 'general', entities = { teams: [], players: [], years: [], venues: [], matchTypes: [], competitions: [], match_id: undefined } } = input;
  const analysis = await analyzeQuery(question);
  logger.info(`🔍 Processing ${analysis.type} query (confidence: ${analysis.confidence.toFixed(2)}) with strategy: ${analysis.searchStrategy.join(', ')}`);

  const contexts: Context[] = [];
  const metadata: any = {
    sources: [],
    processingTime: {},
    resultCounts: {},
    errors: [],
  };

  // Match ID query
  if (entities.match_id) {
    try {
      const startTime = Date.now();
      const results = await getMatchById(entities.match_id);
      if (results.length > 0) {
        contexts.push(...results);
        metadata.sources.push('sql_match_id');
        metadata.processingTime.sql_match_id = Date.now() - startTime;
        metadata.resultCounts.sql_match_id = results[0].matches.length;
      }
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown match ID error';
      logger.error(`❌ Match ID context error: ${errorMessage}`);
      metadata.errors.push(`sql_match_id: ${errorMessage}`);
    }
  }

  // SQL searches
  const sqlSearches = [
    { strategy: 'sql_specific', handler: getSpecificSQLContext },
    { strategy: 'sql_stats', handler: getStatsContext },
    { strategy: 'sql_recent', handler: getRecentContext },
    { strategy: 'sql_championship', handler: getChampionshipContext },
    { strategy: 'sql_venue', handler: getVenueContext },
    { strategy: 'sql_score', handler: getScoreContext },
  ];

  for (const search of sqlSearches) {
    if (analysis.searchStrategy.includes(search.strategy)) {
      let retries = 2;
      while (retries > 0) {
        try {
          const sqlStart = Date.now();
          const sqlResults = await Promise.race([
            search.handler(analysis.entities, question),
            new Promise((_, reject) => setTimeout(() => reject(new Error(`${search.strategy} timeout`)), 5000)),
          ]) as any[];
          if (sqlResults.length > 0) {
            contexts.push(...sqlResults);
            metadata.sources.push(`postgres_${search.strategy}`);
            metadata.processingTime[search.strategy] = Date.now() - sqlStart;
            metadata.resultCounts[search.strategy] = sqlResults.length;
          }
          break;
        } catch (error: unknown) {
          retries--;
          const errorMessage = error instanceof Error ? error.message : 'Unknown SQL error';
          logger.warn(`⚠️ ${search.strategy} failed (retries left: ${retries}): ${errorMessage}`);
          if (retries === 0) {
            metadata.errors.push(`${search.strategy}: ${errorMessage}`);
          }
          if (retries > 0) {
            await new Promise(resolve => setTimeout(resolve, 1000));
          }
        }
      }
    }
  }

  // Player-specific enhanced search
  if (analysis.entities.players.length > 0) {
    let retries = 2;
    while (retries > 0) {
      try {
        const playerStart = Date.now();
        const playerResults = await Promise.race([
          getEnhancedPlayerContext(analysis.entities.players, analysis.entities.years),
          new Promise((_, reject) => setTimeout(() => reject(new Error('Player search timeout')), 5000)),
        ]) as any[];
        if (playerResults.length > 0) {
          contexts.push(...playerResults);
          metadata.sources.push('postgres_players_enhanced');
          metadata.processingTime.player_enhanced = Date.now() - playerStart;
          metadata.resultCounts.player_enhanced = playerResults.length;
        }
        break;
      } catch (error: unknown) {
        retries--;
        const errorMessage = error instanceof Error ? error.message : 'Unknown player error';
        logger.warn(`⚠️ Enhanced player search failed (retries left: ${retries}): ${errorMessage}`);
        if (retries === 0) {
          metadata.errors.push(`Player enhanced: ${errorMessage}`);
        }
        if (retries > 0) {
          await new Promise(resolve => setTimeout(resolve, 1000));
        }
      }
    }
  }

  // Vector search only for general queries without match_id
  if (analysis.needsVector) {
    let retries = 2;
    while (retries > 0) {
      try {
        const vectorStart = Date.now();
        const store = await Promise.race([
          getVectorStore(),
          new Promise((_, reject) => setTimeout(() => reject(new Error('Vector store timeout')), 5000)),
        ]);
        const searchTerms = enhanceSearchTerms(question, analysis.entities);
        const vectorResults = await Promise.race([
          (store as QdrantVectorStore).similaritySearch(searchTerms, 5),
          new Promise((_, reject) => setTimeout(() => reject(new Error('Vector search timeout')), 5000)),
        ]) as Array<any>;

        if (vectorResults.length > 0) {
          contexts.push({
            source: 'vector_search',
            title: 'Similar Matches Found',
            data: vectorResults.map((r, i) => ({
              type: 'match_summary',
              content: r.pageContent,
              relevance: i === 0 ? 'high' : i < 3 ? 'medium' : 'low',
              score: r.metadata?.score || 1.0,
            })),
          });
          metadata.sources.push('qdrant');
          metadata.processingTime.vector = Date.now() - vectorStart;
          metadata.resultCounts.vector = vectorResults.length;
        }
        break;
      } catch (error: unknown) {
        retries--;
        const errorMessage = error instanceof Error ? error.message : 'Unknown vector error';
        logger.warn(`⚠️ Vector search failed (retries left: ${retries}): ${errorMessage}`);
        if (retries === 0) {
          metadata.errors.push(`Vector search: ${errorMessage}`);
        }
        if (retries > 0) {
          await new Promise(resolve => setTimeout(resolve, 1000));
        }
      }
    }
  }

  const combinedContext = formatCombinedContextForMistral(contexts, analysis);
  logger.info(`📊 Context generated from ${metadata.sources.length} sources: ${metadata.sources.join(', ')}`);
  return combinedContext;
}

// Enhanced search terms
function enhanceSearchTerms(question: string, entities: Entities): string {
  let enhancedTerms = question;
  if (!question.toLowerCase().includes('cricket') && !question.toLowerCase().includes('psl')) {
    enhancedTerms += ' cricket PSL';
  }
  if (entities.teams.length > 0) {
    enhancedTerms += ` ${entities.teams.join(' ')}`;
  }
  if (entities.players.length > 0) {
    enhancedTerms += ` ${entities.players.join(' ')}`;
  }
  if (entities.competitions.some(c => c.startsWith('score:'))) {
    const scoreMatch = entities.competitions.find(c => c.startsWith('score:'));
    if (scoreMatch) {
      const score = scoreMatch.replace('score:', '');
      enhancedTerms += ` ${score} runs`;
    }
  }
  return enhancedTerms;
}

// Match ID query
async function getMatchById(matchId: string): Promise<any[]> {
  let client: PoolClient | undefined;
  try {
    client = await postgres.connect();
    const query = `
      SELECT 
        m.match_id, m.team1, m.team2, m.match_date, m.venue, m.winner,
        m.event_name, m.season, m.event_stage,
        i1.total_runs as team1_runs, i1.total_wickets as team1_wickets,
        i2.total_runs as team2_runs, i2.total_wickets as team2_wickets,
        m.player_of_match, m.toss_winner, m.toss_decision,
        m.win_by_runs, m.win_by_wickets
      FROM matches m
      LEFT JOIN innings_summary i1 ON m.match_id = i1.match_id AND i1.innings_number = 1
      LEFT JOIN innings_summary i2 ON m.match_id = i2.match_id AND i2.innings_number = 2
      WHERE m.match_id = $1
    `;
    const result = await client.query(query, [matchId]);
    return [{ type: 'specific_match', match_id: matchId, matches: result.rows }];
  } catch (error: unknown) {
    logger.error(`❌ Match ID query error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    return [];
  } finally {
    if (client) client.release();
  }
}

// Team performance metrics
async function getTeamPerformanceMetrics(teamName: string, seasonLimit: number = 0) {
  let client: PoolClient | undefined;
  try {
    client = await postgres.connect();
    const teamMetricsQuery = `
      SELECT 
        $1 as team_name,
        COUNT(*) as total_matches,
        SUM(CASE WHEN m.winner = $1 THEN 1 ELSE 0 END) as wins,
        SUM(CASE WHEN m.winner != $1 AND m.winner IS NOT NULL THEN 1 ELSE 0 END) as losses,
        SUM(CASE WHEN m.winner IS NULL THEN 1 ELSE 0 END) as draws,
        ROUND(AVG(CASE WHEN m.team1 = $1 THEN i1.total_runs ELSE i2.total_runs END), 1) as avg_runs_scored,
        ROUND(AVG(CASE WHEN m.team1 = $1 THEN i2.total_runs ELSE i1.total_runs END), 1) as avg_runs_conceded,
        ROUND(AVG(CASE WHEN m.team1 = $1 THEN i1.run_rate ELSE i2.run_rate END), 2) as avg_run_rate,
        ROUND(AVG(CASE WHEN m.team1 = $1 THEN i1.total_wickets ELSE i2.total_wickets END), 1) as avg_wickets_lost,
        MAX(CASE WHEN m.team1 = $1 THEN i1.total_runs ELSE i2.total_runs END) as highest_score,
        MIN(CASE WHEN m.team1 = $1 THEN i1.total_runs ELSE i2.total_runs END) as lowest_score,
        COUNT(DISTINCT m.venue) as venues_played,
        COUNT(DISTINCT CASE WHEN m.team1 = $1 THEN m.team2 ELSE m.team1 END) as opponents_faced,
        ROUND(SUM(CASE WHEN m.winner = $1 THEN 1 ELSE 0 END)::decimal / COUNT(*)::decimal * 100, 1) as win_percentage
      FROM matches m
      LEFT JOIN innings_summary i1 ON m.match_id = i1.match_id AND i1.innings_number = 1
      LEFT JOIN innings_summary i2 ON m.match_id = i2.match_id AND i2.innings_number = 2
      WHERE (m.team1 = $1 OR m.team2 = $1)
      AND m.match_date IS NOT NULL
      ${seasonLimit > 0 ? 'AND m.season >= (SELECT MAX(season)::integer - $2 FROM matches)' : ''}
      ORDER BY m.match_date DESC
    `;
    const params = seasonLimit > 0 ? [teamName, seasonLimit] : [teamName];
    const result = await client.query(teamMetricsQuery, params);
    return result.rows[0] || null;
  } finally {
    if (client) client.release();
  }
}

// Score context
async function getScoreContext(entities: Entities, question: string): Promise<any[]> {
  let client: PoolClient | undefined;
  const results: any[] = [];
  try {
    client = await postgres.connect();
    const score = entities.competitions.find(c => c.startsWith('score:'))?.replace('score:', '');
    if (score) {
      const scoreQuery = `
        SELECT 
          m.match_id, m.team1, m.team2, m.match_date, m.venue, m.winner,
          m.event_name, m.season, m.event_stage,
          i1.total_runs as team1_runs, i1.total_wickets as team1_wickets,
          i2.total_runs as team2_runs, i2.total_wickets as team2_wickets,
          m.player_of_match, m.toss_winner, m.toss_decision
        FROM matches m
        LEFT JOIN innings_summary i1 ON m.match_id = i1.match_id AND i1.innings_number = 1
        LEFT JOIN innings_summary i2 ON m.match_id = i2.match_id AND i2.innings_number = 2
        WHERE i1.total_runs = $1 OR i2.total_runs = $1
        ORDER BY m.match_date DESC
      `;
      const result = await client.query(scoreQuery, [parseInt(score)]);
      if (result.rows.length > 0) {
        results.push({
          type: 'score_matches',
          score,
          matches: result.rows,
        });
      }
    }
  } catch (error: unknown) {
    logger.error(`❌ Score context error: ${error instanceof Error ? error.message : 'Unknown error'}`);
  } finally {
    if (client) client.release();
  }
  return results;
}

// Enhanced player context with season filter
async function getEnhancedPlayerContext(players: string[], years: string[] = []): Promise<any[]> {
  let client: PoolClient | undefined;
  const results: any[] = [];
  try {
    client = await postgres.connect();
    for (const player of players) {
      const playerQuery = `
        SELECT 
          ps.player_name, ps.team_name, m.season,
          COUNT(DISTINCT ps.match_id) as matches_played,
          COALESCE(SUM(ps.runs_scored), 0) as total_runs,
          COALESCE(SUM(ps.balls_faced), 0) as total_balls_faced,
          COALESCE(SUM(ps.wickets_taken), 0) as total_wickets,
          COALESCE(SUM(ps.overs_bowled), 0) as total_overs,
          COALESCE(SUM(ps.runs_conceded), 0) as total_runs_conceded,
          ROUND(AVG(NULLIF(ps.strike_rate, 0)), 1) as avg_strike_rate,
          ROUND(AVG(NULLIF(ps.economy_rate, 0)), 2) as avg_economy_rate,
          COALESCE(SUM(ps.fours), 0) as total_fours,
          COALESCE(SUM(ps.sixes), 0) as total_sixes,
          MAX(ps.runs_scored) as highest_score,
          MAX(ps.wickets_taken) as best_bowling
        FROM player_stats ps
        JOIN matches m ON ps.match_id = m.match_id
        WHERE LOWER(ps.player_name) = LOWER($1)
        ${years.length > 0 ? 'AND m.season = ANY($2)' : ''}
        GROUP BY ps.player_name, ps.team_name, m.season
      `;
      const params = years.length > 0 ? [player, years] : [player];
      const statsResult = await client.query(playerQuery, params);

      const recentPerformanceQuery = `
        SELECT 
          m.match_id, m.team1, m.team2, m.match_date, m.venue, m.winner,
          ps.runs_scored, ps.balls_faced, ps.wickets_taken, ps.overs_bowled,
          ps.strike_rate, ps.economy_rate, ps.fours, ps.sixes,
          ts.team_name as player_team
        FROM matches m
        JOIN team_squads ts ON m.match_id = ts.match_id
        LEFT JOIN player_stats ps ON m.match_id = ps.match_id AND LOWER(ps.player_name) = LOWER($1)
        WHERE LOWER(ts.player_name) = LOWER($1)
        ${years.length > 0 ? 'AND m.season = ANY($2)' : ''}
        ORDER BY m.match_date DESC
      `;
      const performanceResult = await client.query(recentPerformanceQuery, params);

      results.push({
        type: 'enhanced_player_profile',
        player,
        stats: statsResult.rows,
        recentPerformance: performanceResult.rows,
      });
    }
  } catch (error: unknown) {
    logger.error(`❌ Enhanced player context error: ${error instanceof Error ? error.message : 'Unknown error'}`);
  } finally {
    if (client) client.release();
  }
  return results;
}

// Venue context
async function getVenueContext(entities: Entities, question: string): Promise<any[]> {
  let client: PoolClient | undefined;
  const results: any[] = [];
  try {
    client = await postgres.connect();
    if (entities.venues.length > 0) {
      for (const venue of entities.venues) {
        const venueQuery = `
          SELECT 
            venue,
            COUNT(*) as total_matches,
            COUNT(DISTINCT winner) as different_winners,
            COUNT(DISTINCT season) as seasons_hosted,
            ROUND(AVG(COALESCE(i1.total_runs, 0)), 1) as avg_first_innings_runs,
            ROUND(AVG(COALESCE(i2.total_runs, 0)), 1) as avg_second_innings_runs,
            MAX(i1.total_runs) as highest_first_innings,
            MAX(i2.total_runs) as highest_second_innings,
            SUM(CASE WHEN i1.total_runs > i2.total_runs THEN 1 ELSE 0 END) as first_innings_wins,
            SUM(CASE WHEN i2.total_runs > i1.total_runs THEN 1 ELSE 0 END) as second_innings_wins
          FROM matches m
          LEFT JOIN innings_summary i1 ON m.match_id = i1.match_id AND i1.innings_number = 1
          LEFT JOIN innings_summary i2 ON m.match_id = i2.match_id AND i2.innings_number = 2
          WHERE LOWER(m.venue) LIKE LOWER($1)
          AND m.winner IS NOT NULL
          GROUP BY venue
        `;
        const result = await client.query(venueQuery, [`%${venue}%`]);
        if (result.rows.length > 0) {
          results.push({ type: 'venue_statistics', venue, stats: result.rows[0] });
        }
      }
    } else {
      const generalVenueQuery = `
        SELECT 
          venue,
          COUNT(*) as matches_hosted,
          COUNT(DISTINCT winner) as different_winners,
          ROUND(AVG(COALESCE(i1.total_runs, 0)), 1) as avg_runs_first_innings
        FROM matches m
        LEFT JOIN innings_summary i1 ON m.match_id = i1.match_id AND i1.innings_number = 1
        WHERE m.venue IS NOT NULL AND m.venue != ''
        GROUP BY venue
        ORDER BY matches_hosted DESC
      `;
      const result = await client.query(generalVenueQuery);
      if (result.rows.length > 0) {
        results.push({ type: 'general_venue_stats', venues: result.rows });
      }
    }
  } catch (error: unknown) {
    logger.error(`❌ Venue context error: ${error instanceof Error ? error.message : 'Unknown error'}`);
  } finally {
    if (client) client.release();
  }
  return results;
}

// Specific SQL context
async function getSpecificSQLContext(entities: Entities, question: string): Promise<any[]> {
  let client: PoolClient | undefined;
  const results: any[] = [];
  try {
    client = await postgres.connect();
    if (entities.teams.length > 0) {
      const teamQuery = `
        SELECT 
          m.match_id, m.team1, m.team2, m.match_date, m.venue, m.winner,
          m.event_name, m.season, m.event_stage,
          i1.total_runs as team1_runs, i1.total_wickets as team1_wickets,
          i2.total_runs as team2_runs, i2.total_wickets as team2_wickets,
          m.player_of_match, m.toss_winner, m.toss_decision,
          m.win_by_runs, m.win_by_wickets
        FROM matches m
        LEFT JOIN innings_summary i1 ON m.match_id = i1.match_id AND i1.innings_number = 1
        LEFT JOIN innings_summary i2 ON m.match_id = i2.match_id AND i2.innings_number = 2
        WHERE (LOWER(m.team1) = ANY($1) OR LOWER(m.team2) = ANY($1))
        ${entities.years.length > 0 ? 'AND m.season = ANY($2)' : ''}
        ORDER BY m.match_date DESC
      `;
      const queryParams = entities.years.length > 0 ? [entities.teams.map(t => t.toLowerCase()), entities.years] : [entities.teams.map(t => t.toLowerCase())];
      const result = await client.query(teamQuery, queryParams);
      results.push({ type: 'team_matches_enhanced', teams: entities.teams, matches: result.rows });
    }

    if (entities.players.length > 0) {
      const playerMatchesQuery = `
        SELECT DISTINCT
          m.match_id, m.team1, m.team2, m.match_date, m.venue, m.winner,
          ts.team_name as player_team, m.event_name, m.season,
          ps.runs_scored, ps.balls_faced, ps.wickets_taken, ps.strike_rate, ps.economy_rate
        FROM matches m
        JOIN team_squads ts ON m.match_id = ts.match_id
        LEFT JOIN player_stats ps ON m.match_id = ps.match_id AND LOWER(ps.player_name) = ANY($1)
        WHERE LOWER(ts.player_name) = ANY($1)
        ${entities.years.length > 0 ? 'AND m.season = ANY($2)' : ''}
        ORDER BY m.match_date DESC
      `;
      const queryParams = entities.years.length > 0 ? [entities.players.map(p => p.toLowerCase()), entities.years] : [entities.players.map(p => p.toLowerCase())];
      const result = await client.query(playerMatchesQuery, queryParams);
      if (result.rows.length > 0) {
        results.push({ type: 'player_match_history', players: entities.players, matches: result.rows });
      }
    }

    if (entities.matchTypes.length > 0) {
      const matchTypeQuery = `
        SELECT 
          m.match_id, m.team1, m.team2, m.match_date, m.venue, m.winner,
          m.event_name, m.season, m.event_stage,
          i1.total_runs as team1_runs, i1.total_wickets as team1_wickets,
          i2.total_runs as team2_runs, i2.total_wickets as team2_wickets,
          m.player_of_match, m.win_by_runs, m.win_by_wickets
        FROM matches m
        LEFT JOIN innings_summary i1 ON m.match_id = i1.match_id AND i1.innings_number = 1
        LEFT JOIN innings_summary i2 ON m.match_id = i2.match_id AND i2.innings_number = 2
        WHERE LOWER(m.event_stage) = ANY($1)
        ORDER BY m.match_date DESC
      `;
      const result = await client.query(matchTypeQuery, [entities.matchTypes.map(t => t.toLowerCase())]);
      results.push({ type: 'match_type_enhanced', matchTypes: entities.matchTypes, matches: result.rows });
    }

    if (entities.teams.length === 2) {
      const h2hQuery = `
        SELECT 
          m.match_date, m.venue, m.season, m.event_stage,
          m.team1, m.team2, m.winner,
          i1.total_runs as team1_runs, i1.total_wickets as team1_wickets,
          i2.total_runs as team2_runs, i2.total_wickets as team2_wickets,
          m.player_of_match, m.toss_winner
        FROM matches m
        LEFT JOIN innings_summary i1 ON m.match_id = i1.match_id AND i1.innings_number = 1
        LEFT JOIN innings_summary i2 ON m.match_id = i2.match_id AND i2.innings_number = 2
        WHERE ((LOWER(m.team1) = LOWER($1) AND LOWER(m.team2) = LOWER($2)) 
               OR (LOWER(m.team1) = LOWER($2) AND LOWER(m.team2) = LOWER($1)))
        AND m.winner IS NOT NULL
        ORDER BY m.match_date DESC
      `;
      const result = await client.query(h2hQuery, [entities.teams[0], entities.teams[1]]);
      if (result.rows.length > 0) {
        results.push({ type: 'head_to_head', team1: entities.teams[0], team2: entities.teams[1], matches: result.rows });
      }
    }
  } catch (error: unknown) {
    logger.error(`❌ Specific SQL context error: ${error instanceof Error ? error.message : 'Unknown error'}`);
  } finally {
    if (client) client.release();
  }
  return results;
}

// Stats context
async function getStatsContext(entities: Entities, question: string): Promise<any[]> {
  let client: PoolClient | undefined;
  const results: any[] = [];
  try {
    client = await postgres.connect();
    if (entities.teams.length > 0) {
      for (const team of entities.teams) {
        const metrics = await getTeamPerformanceMetrics(team);
        if (metrics) {
          results.push({ type: 'team_statistics', team, stats: metrics });
        }
      }
    }
    if (entities.teams.length === 0 && entities.players.length === 0) {
      const overallStatsQuery = `
        SELECT 
          COUNT(DISTINCT match_id) as total_matches,
          COUNT(DISTINCT winner) as teams_won,
          COUNT(DISTINCT season) as total_seasons,
          MAX(match_date) as latest_match,
          MIN(match_date) as first_match,
          COUNT(DISTINCT venue) as venues_used
        FROM matches
        WHERE winner IS NOT NULL
      `;
      const result = await client.query(overallStatsQuery);
      results.push({ type: 'overall_psl_stats', stats: result.rows[0] });
    }
  } catch (error: unknown) {
    logger.error(`❌ Stats context error: ${error instanceof Error ? error.message : 'Unknown error'}`);
  } finally {
    if (client) client.release();
  }
  return results;
}

// Recent matches context
async function getRecentContext(entities: Entities, question: string): Promise<any[]> {
  let client: PoolClient | undefined;
  const results: any[] = [];
  try {
    client = await postgres.connect();
    let recentQuery = `
      SELECT 
        m.match_id, m.team1, m.team2, m.match_date, m.venue, m.winner,
        m.event_name, m.season, m.event_stage,
        i1.total_runs as team1_runs, i1.total_wickets as team1_wickets,
        i2.total_runs as team2_runs, i2.total_wickets as team2_wickets,
        m.player_of_match, m.toss_winner, m.toss_decision
      FROM matches m
      LEFT JOIN innings_summary i1 ON m.match_id = i1.match_id AND i1.innings_number = 1
      LEFT JOIN innings_summary i2 ON m.match_id = i2.match_id AND i2.innings_number = 2
      WHERE m.match_date IS NOT NULL
    `;
    const queryParams: any[] = [];
    if (entities.teams.length > 0) {
      recentQuery += ` AND (LOWER(m.team1) = ANY($1) OR LOWER(m.team2) = ANY($1))`;
      queryParams.push(entities.teams.map(t => t.toLowerCase()));
    }
    recentQuery += ` ORDER BY m.match_date DESC`;
    const result = await client.query(recentQuery, queryParams);
    if (result.rows.length > 0) {
      results.push({
        type: 'recent_matches',
        matches: result.rows,
        filter: entities.teams.length > 0 ? `Teams: ${entities.teams.join(', ')}` : 'All teams',
      });
    }
  } catch (error: unknown) {
    logger.error(`❌ Recent context error: ${error instanceof Error ? error.message : 'Unknown error'}`);
  } finally {
    if (client) client.release();
  }
  return results;
}

// Championship context
async function getChampionshipContext(entities: Entities, question: string): Promise<any[]> {
  let client: PoolClient | undefined;
  const results: any[] = [];
  try {
    client = await postgres.connect();
    const championQuery = `
      SELECT 
        m.season,
        m.winner,
        m.event_stage,
        m.match_date,
        m.venue,
        m.team1,
        m.team2,
        i1.total_runs as team1_runs,
        i1.total_wickets as team1_wickets,
        i2.total_runs as team2_runs,
        i2.total_wickets as team2_wickets,
        m.player_of_match
      FROM matches m
      LEFT JOIN innings_summary i1 ON m.match_id = i1.match_id AND i1.innings_number = 1
      LEFT JOIN innings_summary i2 ON m.match_id = i2.match_id AND i2.innings_number = 2
      WHERE (LOWER(m.event_stage) LIKE '%final%' 
             OR LOWER(m.event_stage) LIKE '%champion%'
             OR LOWER(m.event_stage) = 'final')
      AND m.winner IS NOT NULL
      ORDER BY m.match_date DESC
    `;
    const result = await client.query(championQuery);
    if (result.rows.length > 0) {
      results.push({ type: 'championship_history', finals: result.rows });
    }
  } catch (error: unknown) {
    logger.error(`❌ Championship context error: ${error instanceof Error ? error.message : 'Unknown error'}`);
  } finally {
    if (client) client.release();
  }
  return results;
}

// Simplified context formatting
function formatCombinedContextForMistral(contexts: Context[], analysis: QueryAnalysis): string {
  if (contexts.length === 0) {
    return 'No relevant data found in the database.';
  }
  const sections: string[] = [];

  contexts.forEach(context => {
    if (context.source === 'sql_match_id') {
      const match = context.data.find((d: any) => d.type === 'specific_match')?.matches[0];
      if (match) {
        sections.push(`Match ${match.match_id}: ${match.team1}: ${formatScore(match.team1_runs, match.team1_wickets)} vs ${match.team2}: ${formatScore(match.team2_runs, match.team2_wickets)}, Winner: ${match.winner || 'Unknown'}, Venue: ${match.venue}, Date: ${match.match_date}`);
      }
    } else if (context.source === 'player_enhanced') {
      context.data.forEach((playerData: any) => {
        sections.push(`${playerData.player} Stats:`);
        playerData.stats.forEach((stat: any) => {
          sections.push(`- Season: ${stat.season || 'All'}, Matches: ${stat.matches_played}, Runs: ${stat.total_runs}, Wickets: ${stat.total_wickets}, Strike Rate: ${stat.avg_strike_rate || 'N/A'}`);
        });
        if (playerData.recentPerformance?.length > 0) {
          sections.push(`Recent Performances:`);
          playerData.recentPerformance.forEach((perf: any, idx: number) => {
            sections.push(`  [${idx + 1}] ${perf.team1} vs ${perf.team2} (${perf.match_date})`);
            if (perf.runs_scored !== null) {
              sections.push(`  Batting: ${perf.runs_scored} runs (${perf.balls_faced} balls, SR: ${perf.strike_rate || 'N/A'})`);
            }
            if (perf.wickets_taken !== null && perf.wickets_taken > 0) {
              sections.push(`  Bowling: ${perf.wickets_taken} wickets (ER: ${perf.economy_rate || 'N/A'})`);
            }
          });
        }
      });
    } else if (context.source === 'sql_specific') {
      context.data.forEach((dataset: any) => {
        if (dataset.type === 'team_matches_enhanced') {
          sections.push(`${dataset.teams.join(', ')} Matches:`);
          dataset.matches.forEach((match: any, idx: number) => {
            sections.push(`  [${idx + 1}] ${match.team1}: ${formatScore(match.team1_runs, match.team1_wickets)} vs ${match.team2}: ${formatScore(match.team2_runs, match.team2_wickets)}, Winner: ${match.winner || 'Unknown'}, Date: ${match.match_date}`);
          });
        } else if (dataset.type === 'player_match_history') {
          sections.push(`${dataset.players.join(', ')} Match History:`);
          dataset.matches.forEach((match: any, idx: number) => {
            sections.push(`  [${idx + 1}] ${match.team1} vs ${match.team2} (${match.match_date})`);
            if (match.runs_scored !== null) {
              sections.push(`  Batting: ${match.runs_scored} runs`);
            }
            if (match.wickets_taken !== null && match.wickets_taken > 0) {
              sections.push(`  Bowling: ${match.wickets_taken} wickets`);
            }
          });
        }
      });
    } else if (context.source === 'sql_stats') {
      context.data.forEach((dataset: any) => {
        if (dataset.type === 'team_statistics') {
          const stats = dataset.stats;
          sections.push(`${dataset.team} Stats:`);
          sections.push(`- Matches: ${stats.total_matches}, Wins: ${stats.wins}, Win %: ${stats.win_percentage}%`);
          sections.push(`- Avg Runs Scored: ${stats.avg_runs_scored}, Avg Run Rate: ${stats.avg_run_rate}`);
        }
      });
    } else if (context.source === 'vector_search') {
      sections.push('Similar Matches:');
      context.data.forEach((item: any, idx: number) => {
        sections.push(`  [${idx + 1}] ${item.content}`);
      });
    }
  });

  return sections.join('\n');
}

// Score formatting
function formatScore(runs: any, wickets: any): string {
  if (runs === null || runs === undefined) {
    return 'No score recorded';
  }
  const runsStr = runs.toString();
  const wicketsStr = wickets !== null && wickets !== undefined ? `/${wickets}` : '';
  return `${runsStr}${wicketsStr}`;
}

// LLM prompt template
const enhancedTemplate = PromptTemplate.fromTemplate(`
You are an expert PSL (Pakistan Super League) cricket analyst with access to comprehensive match databases. Your role is to provide accurate, data-driven responses about cricket matches, player statistics, and team performance.

CRITICAL INSTRUCTIONS:
1. Use ONLY the information provided in the DATABASE CONTEXT below
2. Do NOT add any external cricket knowledge or assumptions
3. If specific data is not in the context, clearly state: "This information is not available in the current database"
4. Present statistics and match details exactly as provided
5. Format responses clearly with proper structure for cricket fans

DATABASE CONTEXT:
{context}

USER QUESTION:
{question}

RESPONSE REQUIREMENTS:
- Base your answer exclusively on the DATABASE CONTEXT above
- Use specific data points, scores, dates, and statistics from the context
- Format match results as: "Team A: score vs Team B: score, Winner: Team, Venue: Venue, Date: Date"
- For stats, use bullet points (e.g., "• Runs: X", "• Wickets: Y")
- If no data, state: "No relevant data found in the database"

Your response:`);

// Chain creation
export function getChain() {
  return RunnableSequence.from([
    {
      question: (input: { question: string }) => input.question,
      context: async (input: { question: string }) => {
        try {
          return await getEnhancedContext(input);
        } catch (error: unknown) {
          const errorMessage = error instanceof Error ? error.message : 'Unknown context error';
          logger.error(`❌ Context retrieval error: ${errorMessage}`);
          return 'No relevant data found in the database.';
        }
      },
    },
    enhancedTemplate,
    llm,
    new StringOutputParser(),
  ]);
}

// Health check
export async function healthCheck() {
  const checks = {
    postgres: false,
    qdrant: false,
    togetherai: false,
    embeddings: false,
    stats: {},
    responseTime: 0,
  };
  const startTime = Date.now();

  try {
    const client = await postgres.connect();
    await client.query('SELECT 1');
    checks.postgres = true;
    const statsQuery = `
      SELECT 
        (SELECT COUNT(*) FROM matches) as total_matches,
        (SELECT COUNT(*) FROM team_squads) as total_player_records,
        (SELECT COUNT(DISTINCT player_name) FROM player_stats) as unique_players,
        (SELECT MAX(match_date) FROM matches WHERE match_date IS NOT NULL) as latest_match,
        (SELECT MIN(match_date) FROM matches WHERE match_date IS NOT NULL) as earliest_match,
        (SELECT COUNT(DISTINCT venue) FROM matches WHERE venue IS NOT NULL) as total_venues
    `;
    const statsResult = await client.query(statsQuery);
    checks.stats = statsResult.rows[0];
    client.release();
  } catch (error: unknown) {
    logger.error(`❌ Postgres health check failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }

  try {
    await qdrant.getCollections();
    checks.qdrant = true;
    const collectionInfo = await qdrant.getCollection('match_summaries');
    (checks.stats as any).vector_count = collectionInfo.points_count || 0;
  } catch (error: unknown) {
    logger.error(`❌ Qdrant health check failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }

  try {
    await llm.invoke('Test connection');
    checks.togetherai = true;
  } catch (error: unknown) {
    logger.error(`❌ Together AI health check failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }

  try {
    await embeddings.embedQuery('test');
    checks.embeddings = true;
  } catch (error: unknown) {
    logger.error(`❌ Embeddings health check failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }

  checks.responseTime = Date.now() - startTime;
  return checks;
}

// Utility functions
export async function testChain(question: string) {
  try {
    const chain = getChain();
    return await chain.invoke({ question });
  } catch (error: unknown) {
    logger.error(`❌ Chain test error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    throw error;
  }
}

export async function testContext(question: string) {
  try {
    const result = await getEnhancedContext({ question });
    return {
      context: result,
      metadata: { sources: ['combined'], message: 'Enhanced context retrieval' },
    };
  } catch (error: unknown) {
    logger.error(`❌ Context test error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    throw error;
  }
}

export async function getPlayerStats(playerName: string) {
  let client: PoolClient | undefined;
  try {
    client = await postgres.connect();
    const playerQuery = `
      SELECT 
        ps.player_name, ps.team_name,
        COUNT(DISTINCT ps.match_id) as matches_played,
        COALESCE(SUM(ps.runs_scored), 0) as total_runs,
        COALESCE(SUM(ps.balls_faced), 0) as total_balls_faced,
        COALESCE(SUM(ps.wickets_taken), 0) as total_wickets,
        ROUND(AVG(NULLIF(ps.strike_rate, 0)), 1) as avg_strike_rate,
        ROUND(AVG(NULLIF(ps.economy_rate, 0)), 2) as avg_economy_rate,
        COALESCE(SUM(ps.fours), 0) as total_fours,
        COALESCE(SUM(ps.sixes), 0) as total_sixes,
        MAX(ps.runs_scored) as highest_score
      FROM player_stats ps
      WHERE LOWER(ps.player_name) = LOWER($1)
      GROUP BY ps.player_name, ps.team_name
      ORDER BY total_runs DESC
    `;
    const result = await client.query(playerQuery, [playerName]);
    return result.rows;
  } catch (error: unknown) {
    logger.error(`❌ Player stats error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    return [];
  } finally {
    if (client) client.release();
  }
}

export { getEnhancedContext, postgres, getTeamPerformanceMetrics };