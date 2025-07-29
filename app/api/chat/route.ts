// app/api/chat/route.ts - Enhanced with Together AI Mistral integration
import { streamText } from 'ai';
import { createOpenAI } from '@ai-sdk/openai';
import { getChain, analyzeQuery, healthCheck } from '@/lib/chain';
import { CoreMessage } from 'ai';

// Initialize Together AI with Mistral
const togetherAI = createOpenAI({
  baseURL: 'https://api.together.xyz/v1',
  apiKey: process.env.TOGETHERAI_API_KEY,
});

export async function POST(req: Request) {
  try {
    console.log('🚀 Chat API called with Together AI Mistral');

    const body = await req.json();
    const { messages }: { messages: CoreMessage[] } = body;

    if (!messages || messages.length === 0) {
      return new Response(
        JSON.stringify({ error: 'No messages provided' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      );
    }

    const userMessage = messages[messages.length - 1];
    const input = typeof userMessage.content === 'string' ? userMessage.content : '';

    if (!input.trim()) {
      return new Response(
        JSON.stringify({ error: 'Empty message content' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      );
    }

    console.log('📝 User question:', input);

    // Analyze the query to understand what type of response is needed
    const queryAnalysis = await analyzeQuery(input);
    console.log('🔍 Query analysis:', queryAnalysis);

    // Get enhanced context from chain (combines vector + SQL search)
    const chain = getChain();
    let chainResult: string;
    let contextMetadata: any = {};

    try {
      const startTime = Date.now();
      const result = await chain.invoke({ question: input });
      
      if (!result || result.includes("No specific match data found")) {
        return new Response(
          JSON.stringify({
            type: "info",
            message: "I don't have that specific record in the current PSL database. Please try asking about teams, players, or matches that are included in our PSL dataset.",
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      
      const processingTime = Date.now() - startTime;
      console.log(`⚡ Chain processing completed in ${processingTime}ms`);
      
      // Handle both string and object responses
      if (typeof result === 'string') {
        chainResult = result;
      } else if (result && typeof result === 'object') {
        chainResult = (result as any).context || (result as any).content || String(result);
        contextMetadata = (result as any).metadata || {};
      } else {
        chainResult = String(result);
      }
      
      console.log('🔗 Chain result length:', chainResult?.length || 0);
      console.log('📊 Context metadata:', contextMetadata);
      
    } catch (chainError) {
      console.error('❌ Chain error:', chainError);
      chainResult = 'I apologize, but I encountered an issue accessing the cricket database. I can still help with general PSL questions based on my knowledge.';
      contextMetadata = { error: true, fallback: true };
    }

    // Create context-aware system message based on query type and available data
    const systemMessage = createSystemMessage(input, chainResult, queryAnalysis, contextMetadata);

    // Use Together AI Mistral model with optimized parameters
    const result = await streamText({
      model: togetherAI('mistralai/Mistral-7B-Instruct-v0.2'),
      messages: [systemMessage, userMessage],
      temperature: queryAnalysis.type === 'stats' ? 0.1 : 0.4, // Lower temps for better accuracy
      maxTokens: 1200,
      topP: 0.9,
      frequencyPenalty: 0.1,
      presencePenalty: 0.1,
    });

    return result.toDataStreamResponse();
    
  } catch (error) {
    console.error('❌ Chat API error:', error);
    return new Response(
      JSON.stringify({
        error: 'Failed to process chat request',
        details: error instanceof Error ? error.message : 'Unknown error',
      }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }
}

// Enhanced system message creation optimized for Mistral
function createSystemMessage(
  userQuestion: string, 
  context: string, 
  queryAnalysis: any, 
  metadata: any
): CoreMessage {
  
  let basePrompt = `You are an expert PSL (Pakistan Super League) cricket analyst with access to comprehensive match databases. You provide accurate, data-driven responses about cricket matches, player statistics, and team performance.

IMPORTANT: You must ONLY use the database context provided below. Do NOT use any external knowledge or make assumptions beyond what's explicitly stated in the context.`;
  
  // Add context information with clear boundaries
  if (context && !metadata.error) {
    basePrompt += `\n\nDATABASE CONTEXT (Use ONLY this information):\n${context}\n\nEND OF DATABASE CONTEXT`;
  }
  
  // Add specific instructions based on query type - optimized for Mistral
  switch (queryAnalysis.type) {
    case 'stats':
      basePrompt += `\n\nTASK: This is a STATISTICS query. Requirements:
• Extract exact numbers from the database context
• Present data in clear, structured format
• Use bullet points or tables for clarity
• Include comparative context when available in the data
• State "Data not available" if specific stats aren't in the context
• Be precise and factual - no estimates or assumptions`;
      break;
      
    case 'match_report':
      basePrompt += `\n\nTASK: This is a MATCH REPORT query. Requirements:
• Tell the match story using only provided match data
• Include specific details: teams, scores, venue, date, winner
• Mention player of the match if available in context
• Describe match flow based on available innings data
• Make it engaging but stick to facts from the database
• If match details are incomplete, mention what data is available`;
      break;
      
    case 'player_info':
      basePrompt += `\n\nTASK: This is a PLAYER INFORMATION query. Requirements:
• Use only player data from the database context
• Include available statistics: runs, wickets, matches, averages
• Mention team associations if provided
• Include recent match performance if available
• State clearly if player data is limited or unavailable
• Do not speculate about player abilities beyond provided stats`;
      break;
      
    case 'team_info':
      basePrompt += `\n\nTASK: This is a TEAM INFORMATION query. Requirements:
• Use only team data from the database context
• Include available match results and statistics
• Mention venues, wins, recent performance from provided data
• Include championship information if available in context
• Be clear about data limitations
• Focus on factual team performance metrics`;
      break;
      
    case 'comparison':
      basePrompt += `\n\nTASK: This is a COMPARISON query. Requirements:
• Compare using only data available in the database context
• Present clear head-to-head statistics if available
• Use structured format (tables/bullet points) for clarity
• Highlight specific differences found in the data
• State if comparison is limited due to insufficient data
• Be objective and data-driven`;
      break;
      
    default:
      basePrompt += `\n\nTASK: This is a GENERAL cricket query. Requirements:
• Respond using only the database context provided
• Be conversational but accurate
• Acknowledge data limitations honestly
• Focus on PSL cricket topics
• Provide helpful information based on available context`;
  }
  
  // Add critical response guidelines for Mistral
  basePrompt += `\n\nCRITICAL RESPONSE RULES:
1. ONLY use information from the DATABASE CONTEXT above
2. If the context doesn't contain requested information, respond: "That specific information is not available in the current database."
3. Never make up statistics, player names, or match details
4. Always cite specific data points from the context
5. Use clear, engaging language appropriate for cricket fans
6. Format statistics and match details clearly
7. If context is limited, explain what information IS available`;
  
  // Handle error states
  if (metadata.error) {
    basePrompt += `\n\nNOTE: Database access is currently limited. Provide the best response possible while acknowledging this limitation clearly to the user.`;
  }
  
  return {
    role: 'system',
    content: basePrompt
  };
}

// Enhanced health check with Together AI status
export async function GET() {
  try {
    const healthStatus = await healthCheck();
    
    return new Response(
      JSON.stringify({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        services: {
          togetherai: !!process.env.TOGETHERAI_API_KEY,
          postgres: healthStatus.postgres,
          qdrant: healthStatus.qdrant,
          embeddings: healthStatus.embeddings,
        },
        database_stats: healthStatus.stats || {},
        response_time: healthStatus.responseTime || 0,
        model: 'mistralai/Mistral-7B-Instruct-v0.2'
      }),
      { 
        status: 200, 
        headers: { 
          'Content-Type': 'application/json',
          'Cache-Control': 'no-cache, no-store, must-revalidate'
        } 
      }
    );
  } catch (error) {
    console.error('❌ Health check error:', error);
    return new Response(
      JSON.stringify({
        status: 'degraded',
        timestamp: new Date().toISOString(),
        error: error instanceof Error ? error.message : 'Unknown error',
        services: {
          togetherai: !!process.env.TOGETHERAI_API_KEY,
          postgres: false,
          qdrant: false,
          embeddings: false,
        }
      }),
      { 
        status: 503,
        headers: { 'Content-Type': 'application/json' }
      }
    );
  }
}