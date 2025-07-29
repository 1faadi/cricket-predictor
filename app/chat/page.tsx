'use client';

import { useChat } from 'ai/react';
import { useRef, useEffect, useState } from 'react';
import { Send, Trophy, Users, Target, Calendar, MapPin, Star, Loader2, AlertCircle } from 'lucide-react';

function cn(...classes: string[]) {
  return classes.filter(Boolean).join(' ');
}

export default function ChatPage() {
    const {
        messages,
        input,
        handleInputChange,
        handleSubmit,
        isLoading,
        error,
      } = useChat({
        api: '/api/chat',
      
        onFinish: (message) => {
          console.log('✅ Full message streamed:', message);
        },
      
        onResponse: (response) => {
          console.log('🟢 Response stream started');
          setDebugInfo(prev => [...prev, `Streaming started. Status: ${response.status}`]);
        },
      
        onError: (error) => {
          console.error('Chat error:', error);
          setDebugInfo(prev => [...prev, `Error: ${error.message}`]);
        },
      });
      

  const bottomRef = useRef<HTMLDivElement | null>(null);
  const [typing, setTyping] = useState(false);
  const [debugInfo, setDebugInfo] = useState<string[]>([]);
  const [showDebug, setShowDebug] = useState(false);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (isLoading) {
      setTyping(true);
    } else {
      const timer = setTimeout(() => setTyping(false), 500);
      return () => clearTimeout(timer);
    }
  }, [isLoading]);

  // Log messages for debugging
  useEffect(() => {
    console.log('Current messages:', messages);
    setDebugInfo(prev => [...prev, `Messages count: ${messages.length}`]);
  }, [messages]);

  // Enhanced message rendering that detects content patterns
  const renderMessage = (content: string) => {
    // Check if content looks like statistics (has bullet points or key-value pairs)
    const hasStats = content.includes('•') || 
                    content.includes('**') || 
                    content.includes('Matches Won') ||
                    content.includes('Average') ||
                    content.includes('Strike Rate') ||
                    content.includes('Runs') ||
                    content.includes('Wickets');

    // Check if content looks like a match report
    const isMatchReport = content.toLowerCase().includes('match') && 
                         (content.toLowerCase().includes('vs') || 
                          content.toLowerCase().includes('against') ||
                          content.toLowerCase().includes('final') ||
                          content.toLowerCase().includes('won') ||
                          content.toLowerCase().includes('lost'));

    if (hasStats) {
      return renderStatsResponse(content);
    } else if (isMatchReport) {
      return renderMatchReport(content);
    } else {
      return renderGeneralResponse(content);
    }
  };

  const renderStatsResponse = (content: string) => {
    return (
      <div className="bg-gradient-to-br from-green-900/30 to-green-800/20 border-2 border-green-600/30 rounded-xl p-4 backdrop-blur-sm">
        <div className="flex items-center gap-2 mb-3">
          <Trophy className="w-5 h-5 text-green-400" />
          <h3 className="text-lg font-bold text-green-100">Statistics</h3>
        </div>
        <div className="text-white/90 leading-relaxed space-y-2">
          {content.split('\n').map((line, idx) => {
            // Format lines that look like stats
            if (line.includes('•') || line.includes('**')) {
              return (
                <div key={idx} className="flex items-center gap-2">
                  <Star className="w-3 h-3 text-green-400 flex-shrink-0" />
                  <span className="text-green-100">{line.replace(/[•*]/g, '').trim()}</span>
                </div>
              );
            }
            return line.trim() ? <div key={idx}>{line}</div> : null;
          }).filter(Boolean)}
        </div>
      </div>
    );
  };

  const renderMatchReport = (content: string) => {
    return (
      <div className="bg-gradient-to-br from-white/10 to-white/5 border-2 border-white/20 rounded-xl p-4 backdrop-blur-sm">
        <div className="flex items-center gap-2 mb-3">
          <Trophy className="w-5 h-5 text-green-400" />
          <h3 className="text-lg font-bold text-green-100">Match Information</h3>
        </div>
        <div className="text-white/90 leading-relaxed whitespace-pre-wrap">{content}</div>
      </div>
    );
  };

  const renderGeneralResponse = (content: string) => {
    return (
      <div className="text-white/90 leading-relaxed whitespace-pre-wrap">
        {content}
      </div>
    );
  };

  const quickPrompts = [
    "Tell me about the latest PSL final",
    "Show me Babar Azam's statistics", 
    "Which team has won the most matches?",
    "Compare Karachi Kings vs Lahore Qalandars"
  ];

  const testAPI = async () => {
    try {
      const response = await fetch('/api/chat', {
        method: 'GET'
      });
      const result = await response.json();
      console.log('API Health Check:', result);
      setDebugInfo(prev => [...prev, `API Health: ${JSON.stringify(result)}`]);
    } catch (error) {
      console.error('API Health Check failed:', error);
      setDebugInfo(prev => [...prev, `API Health Check failed: ${error}`]);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-900 via-green-800 to-green-900 text-white">
      {/* Header */}
      <div className="bg-gradient-to-r from-green-600 to-green-700 border-b-4 border-white shadow-lg">
        <div className="max-w-4xl mx-auto px-4 py-6">
          <div className="flex items-center justify-center gap-3">
            <div className="w-8 h-6 bg-green-600 border-2 border-white rounded-sm flex items-center justify-center relative">
              <div className="w-4 h-4 bg-white rounded-full flex items-center justify-center">
                <Star className="w-2 h-2 text-green-600 fill-current" />
              </div>
            </div>
            <h1 className="text-2xl font-bold text-white">🏏 PSL Cricket Assistant</h1>
            <div className="w-8 h-6 bg-green-600 border-2 border-white rounded-sm flex items-center justify-center relative">
              <div className="w-4 h-4 bg-white rounded-full flex items-center justify-center">
                <Star className="w-2 h-2 text-green-600 fill-current" />
              </div>
            </div>
          </div>
          <p className="text-center text-green-100 mt-2">Your expert guide to Pakistan Super League cricket</p>
          
          {/* Debug Toggle */}
          <div className="flex justify-center gap-2 mt-3">
            <button
              onClick={() => setShowDebug(!showDebug)}
              className="text-xs bg-white/20 px-3 py-1 rounded-full hover:bg-white/30 transition-colors"
            >
              {showDebug ? 'Hide Debug' : 'Show Debug'}
            </button>
            <button
              onClick={testAPI}
              className="text-xs bg-white/20 px-3 py-1 rounded-full hover:bg-white/30 transition-colors"
            >
              Test API
            </button>
          </div>
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-4 py-6 flex flex-col min-h-[calc(100vh-140px)]">
        {/* Debug Panel */}
        {showDebug && (
          <div className="mb-4 p-4 bg-red-900/20 border border-red-500/30 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <AlertCircle className="w-4 h-4 text-red-400" />
              <h3 className="font-bold text-red-300">Debug Information</h3>
            </div>
            <div className="text-xs space-y-1 max-h-32 overflow-y-auto">
              <div>Messages: {messages.length}</div>
              <div>Loading: {isLoading ? 'true' : 'false'}</div>
              <div>Error: {error ? error.message : 'none'}</div>
              <div>Input: "{input}"</div>
              <hr className="border-red-500/30 my-2" />
              {debugInfo.slice(-5).map((info, idx) => (
                <div key={idx} className="text-red-200">{info}</div>
              ))}
            </div>
            <button
              onClick={() => setDebugInfo([])}
              className="mt-2 text-xs bg-red-500/20 px-2 py-1 rounded hover:bg-red-500/30"
            >
              Clear Debug
            </button>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="mb-4 p-4 bg-red-900/30 border-2 border-red-500/50 rounded-xl">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-red-400" />
              <div>
                <div className="font-bold text-red-300">Connection Error</div>
                <div className="text-red-200 text-sm">{error.message}</div>
              </div>
            </div>
          </div>
        )}

        {/* Welcome Screen */}
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center flex-1 space-y-6">
            <div className="text-center space-y-4">
              <div className="text-6xl">🏏</div>
              <h2 className="text-2xl font-bold text-green-100">Welcome to PSL Assistant!</h2>
              <p className="text-white/70 max-w-md">
                Ask me anything about Pakistan Super League matches, player statistics, team performances, and more!
              </p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full max-w-2xl">
              {quickPrompts.map((prompt, idx) => (
                <button
                  key={idx}
                  onClick={() => {
                    handleInputChange({ target: { value: prompt } } as any);
                  }}
                  className="p-3 bg-white/10 hover:bg-white/20 border border-white/20 rounded-lg text-left text-sm transition-all duration-200 hover:scale-105"
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Messages */}
        <div className="flex-1 space-y-4 mb-6">
          {messages.map((m, idx) => (
            <div
              key={idx}
              className={cn(
                'flex w-full',
                m.role === 'user' ? 'justify-end' : 'justify-start'
              )}
            >
              <div
                className={cn(
                  'max-w-[80%] rounded-2xl px-4 py-3 shadow-lg',
                  m.role === 'user'
                    ? 'bg-gradient-to-r from-green-600 to-green-700 text-white border-2 border-green-500/50'
                    : 'bg-gradient-to-br from-white/10 to-white/5 text-white border-2 border-white/20 backdrop-blur-sm'
                )}
              >
                {m.role === 'user' ? (
                  <div className="font-medium">{m.content}</div>
                ) : (
                  <div>
                    {renderMessage(m.content)}
                    {showDebug && (
                      <div className="mt-2 text-xs text-gray-400 border-t border-gray-600 pt-2">
                        Raw: {m.content.slice(0, 100)}...
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}
          
          {/* Typing Indicator */}
          {typing && (
            <div className="flex justify-start">
              <div className="bg-gradient-to-br from-white/10 to-white/5 text-white border-2 border-white/20 backdrop-blur-sm rounded-2xl px-4 py-3 shadow-lg">
                <div className="flex items-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin text-green-400" />
                  <span className="text-white/70">Assistant is thinking...</span>
                </div>
              </div>
            </div>
          )}
          
          <div ref={bottomRef} />
        </div>

        {/* Input Area */}
        <div className="sticky bottom-0 bg-gradient-to-r from-green-800/50 to-green-700/50 backdrop-blur-sm rounded-2xl border-2 border-green-600/30 p-4">
          <div className="flex gap-3">
            <input
              value={input}
              onChange={handleInputChange}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey && !isLoading) {
                  e.preventDefault();
                  handleSubmit(e as any);
                }
              }}
              placeholder="Ask about PSL matches, players, statistics, or get match reports..."
              className="flex-1 px-4 py-3 rounded-xl bg-white/10 border-2 border-white/20 text-white placeholder-white/50 focus:outline-none focus:border-green-400/50 focus:bg-white/15 transition-all"
              disabled={isLoading}
            />
            <button
              onClick={(e) => handleSubmit(e as any)}
              className={cn(
                'px-6 py-3 rounded-xl font-medium transition-all duration-200 flex items-center gap-2',
                isLoading
                  ? 'bg-gray-600 cursor-not-allowed text-gray-300'
                  : 'bg-gradient-to-r from-green-600 to-green-700 hover:from-green-500 hover:to-green-600 text-white shadow-lg hover:shadow-xl hover:scale-105'
              )}
              disabled={isLoading}
            >
              {isLoading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
              <span className="hidden sm:inline">
                {isLoading ? 'Sending...' : 'Send'}
              </span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}