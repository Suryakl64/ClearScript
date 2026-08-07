import React, { useState, useEffect } from 'react';
import { translateFindings, getReports } from '../api/client';
import FindingsCard from './FindingsCard';
import LanguageSwitcher from './LanguageSwitcher';
import ChatInterface from './ChatInterface';
import { Activity, Clock, FileText } from 'lucide-react';

export default function Dashboard({ currentFindings, reportId, onNewUpload }) {
  const [findings, setFindings] = useState(currentFindings);
  const [originalFindings] = useState(currentFindings); // preserve English originals
  const [currentLang, setCurrentLang] = useState('en');
  const [isTranslating, setIsTranslating] = useState(false);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    // Load history
    const fetchHistory = async () => {
      try {
        const res = await getReports();
        if (res.success) {
          setHistory(res.reports);
        }
      } catch (err) {
        console.error("Failed to load history", err);
      }
    };
    fetchHistory();
  }, []);

  const handleLanguageChange = async (langCode) => {
    setCurrentLang(langCode);
    if (langCode === 'en') {
      // Revert to original English explanations
      setFindings(originalFindings);
      return;
    }

    setIsTranslating(true);
    try {
      const res = await translateFindings(originalFindings, langCode);
      if (res.success) {
        setFindings(res.findings);
      }
    } catch (err) {
      console.error("Translation failed", err);
    } finally {
      setIsTranslating(false);
    }
  };

  const highRiskCount = findings.filter(f => ['HIGH', 'LOW', 'ABNORMAL'].includes(f.flag?.toUpperCase())).length;

  return (
    <div className="max-w-7xl mx-auto mt-8 grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
      
      {/* Sidebar: History */}
      <div className="lg:col-span-3 bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden h-[calc(100vh-8rem)] sticky top-8 flex flex-col">
        <div className="p-5 border-b border-slate-200 bg-slate-50 flex justify-between items-center">
          <h3 className="font-semibold text-slate-800 flex items-center gap-2">
            <Clock size={18} className="text-clinical-600" />
            Report History
          </h3>
        </div>
        
        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          <button 
            onClick={onNewUpload}
            className="w-full text-left px-4 py-3 rounded-lg border border-dashed border-clinical-300 text-clinical-700 bg-clinical-50 hover:bg-clinical-100 font-medium text-sm flex items-center justify-center gap-2 transition-colors mb-4"
          >
            + Upload New Report
          </button>

          {history.length === 0 ? (
            <p className="text-sm text-slate-500 text-center py-4">No past reports found.</p>
          ) : (
            history.map((rep) => (
              <div 
                key={rep.report_id} 
                className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                  rep.report_id === reportId 
                    ? 'border-clinical-500 bg-clinical-50 shadow-sm' 
                    : 'border-slate-100 bg-white hover:border-slate-300'
                }`}
              >
                <div className="flex items-start gap-3">
                  <FileText size={16} className={`mt-1 ${rep.report_id === reportId ? 'text-clinical-600' : 'text-slate-400'}`} />
                  <div>
                    <p className={`text-sm font-medium truncate w-40 ${rep.report_id === reportId ? 'text-clinical-800' : 'text-slate-700'}`}>
                      {rep.filename}
                    </p>
                    <p className="text-xs text-slate-500 mt-1">
                      {new Date(rep.upload_time).toLocaleDateString()} • {rep.finding_count} findings
                    </p>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Main Content: Findings & Chat */}
      <div className="lg:col-span-9 grid grid-cols-1 lg:grid-cols-5 gap-8">
        
        {/* Left Column: Findings List */}
        <div className="lg:col-span-3 space-y-6">
          <div className="flex justify-between items-end mb-2">
            <div>
              <h2 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
                <Activity size={24} className="text-clinical-600" />
                Your Findings
              </h2>
              <p className="text-slate-500 text-sm mt-1">
                {findings.length} tests analyzed • {highRiskCount} require attention
              </p>
            </div>
            <LanguageSwitcher 
              currentLang={currentLang}
              onLanguageChange={handleLanguageChange}
              isLoading={isTranslating}
            />
          </div>

          <div className="space-y-4">
            {findings.map((finding, idx) => (
              <FindingsCard key={idx} finding={finding} />
            ))}
          </div>
        </div>

        {/* Right Column: Chat Interface */}
        <div className="lg:col-span-2 sticky top-8">
          <ChatInterface reportId={reportId} />
        </div>
      </div>
    </div>
  );
}
