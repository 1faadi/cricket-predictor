// quick-test.ts - Test Gemini API directly
import { config } from 'dotenv';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';

// Load environment variables
config();

async function testGeminiAPI() {
  console.log('🔑 Testing Gemini API...');
  
  // Check if API key exists
  const apiKey = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
  if (!apiKey) {
    console.error('❌ No API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file');
    return;
  }
  
  console.log(`🔑 API key found: ${apiKey.substring(0, 10)}...`);
  
  try {
    const llm = new ChatGoogleGenerativeAI({
      model: 'gemini-1.5-flash',
      apiKey: apiKey,
      temperature: 0.3,
    });
    
    console.log('🚀 Sending test message to Gemini...');
    const result = await llm.invoke('Say "Hello from Gemini!" if you can read this.');
    console.log('✅ Gemini response:', result.content);
    
  } catch (error) {
    console.error('❌ Gemini API test failed:', error instanceof Error ? error.message : error);
  }
}

testGeminiAPI();