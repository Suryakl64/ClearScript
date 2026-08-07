import React, { useState, useRef, useEffect } from 'react';
import { Send, User, Bot, Loader2, Info } from 'lucide-react';
import { askQuestion } from '../api/client';

export default function ChatInterface({ reportId }) {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: "Hello. I've analyzed your medical report. What questions do you have about your findings?",
      disclaimer: true
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || !reportId || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);

    try {
      const response = await askQuestion(reportId, userMessage);
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: response.answer,
          disclaimer: response.disclaimer,
          fallback: !response.llm_available,
          sources: response.relevant_findings || []
        }
      ]);
    } catch (error) {
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: "I'm sorry, I encountered an error while trying to answer your question. Please try again.",
          error: true
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-[600px] bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
      {/* Header */}
      <div className="bg-slate-50 px-5 py-4 border-b border-slate-200">
        <h3 className="font-semibold text-slate-800 flex items-center gap-2">
          <Bot size={20} className="text-clinical-600" />
          ClearScript AI Assistant
        </h3>
        <p className="text-xs text-slate-500 mt-1">Ask questions about your uploaded report</p>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-5 space-y-6 bg-slate-50/50">
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
              msg.role === 'user' ? 'bg-clinical-600 text-white' : 'bg-slate-200 text-slate-600'
            }`}>
              {msg.role === 'user' ? <User size={16} /> : <Bot size={16} />}
            </div>
            
            <div className={`max-w-[80%] ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
              <div className={`rounded-2xl px-4 py-3 ${
                msg.role === 'user' 
                  ? 'bg-clinical-600 text-white' 
                  : msg.error 
                    ? 'bg-red-50 text-red-700 border border-red-100'
                    : 'bg-white border border-slate-200 text-slate-700 shadow-sm'
              }`}>
                <p className="text-sm whitespace-pre-wrap leading-relaxed">{msg.content}</p>
                
                {/* Fallback Badge */}
                {msg.fallback && (
                  <div className="mt-2 text-xs bg-amber-50 text-amber-700 px-2 py-1 rounded inline-block border border-amber-200">
                    Offline Mode: Showing exact findings
                  </div>
                )}

                {/* Sources / References */}
                {msg.sources && msg.sources.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-slate-100">
                    <p className="text-xs font-semibold text-slate-500 mb-1">Based on:</p>
                    <ul className="text-xs text-slate-600 space-y-1">
                      {msg.sources.slice(0, 2).map((src, sIdx) => (
                        <li key={sIdx} className="flex justify-between items-center bg-slate-50 px-2 py-1 rounded">
                          <span className="truncate mr-2">{src.full_name}</span>
                          <span className="font-semibold">{src.value} {src.unit}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
              
              {/* Disclaimer */}
              {msg.disclaimer && (
                <div className="mt-2 flex items-start gap-1.5 text-[10px] text-slate-400 max-w-xs">
                  <Info size={12} className="flex-shrink-0 mt-0.5" />
                  <span>{msg.disclaimer === true ? "Not medical advice. Consult a doctor." : msg.disclaimer}</span>
                </div>
              )}
            </div>
          </div>
        ))}
        
        {isLoading && (
          <div className="flex gap-4">
            <div className="w-8 h-8 rounded-full bg-slate-200 text-slate-600 flex items-center justify-center">
              <Bot size={16} />
            </div>
            <div className="bg-white border border-slate-200 rounded-2xl px-4 py-3 shadow-sm flex items-center gap-2 text-slate-500 text-sm">
              <Loader2 size={16} className="animate-spin" />
              Thinking...
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="p-4 bg-white border-t border-slate-200">
        <div className="relative flex items-center">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={isLoading || !reportId}
            placeholder={reportId ? "Ask about your results..." : "Please upload a report first..."}
            className="w-full pl-4 pr-12 py-3 bg-slate-50 border border-slate-200 rounded-full text-sm focus:outline-none focus:ring-2 focus:ring-clinical-500 focus:border-transparent disabled:opacity-60"
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading || !reportId}
            className="absolute right-1.5 p-2 bg-clinical-600 text-white rounded-full hover:bg-clinical-700 disabled:bg-slate-300 disabled:cursor-not-allowed transition-colors"
          >
            <Send size={16} />
          </button>
        </div>
      </form>
    </div>
  );
}
