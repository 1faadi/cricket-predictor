// test-chain.ts
import { config } from 'dotenv';
import { testChain, healthCheck } from './lib/chain';

// Load environment variables
config();

async function runTests() {
  console.log('🔍 Testing PSL Chain with Gemini...\n');

  // Health check first
  console.log('🏥 Running health checks...');
  try {
    const health = await healthCheck();
    console.log('Health Status:', health);
    console.log('');
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    console.error('❌ Health check failed:', errorMessage);
  }

  // Test questions
  const testQuestions = [
    "Who won PSL 2023?",
    "Tell me about Karachi Kings recent matches", 
    "What are Babar Azam's PSL stats?",
    "PSL finals history",
    "Latest PSL matches"
  ];

  for (const question of testQuestions) {
    console.log(`❓ Question: "${question}"`);
    try {
      const startTime = Date.now();
      const result = await testChain(question);
      const duration = Date.now() - startTime;
      
      console.log(`✅ Answer (${duration}ms): ${result}\n`);
      console.log('─'.repeat(80) + '\n');
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error(`❌ Error: ${errorMessage}\n`);
    }
  }
}

// Run the tests
runTests().catch((error) => {
  const errorMessage = error instanceof Error ? error.message : 'Unknown error';
  console.error('Test runner failed:', errorMessage);
});