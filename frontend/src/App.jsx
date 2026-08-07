import React, { useState } from 'react';
import { BrowserRouter, Routes, Route, useNavigate } from 'react-router-dom';
import UploadPage from './components/UploadPage';
import Dashboard from './components/Dashboard';

function AppContent() {
  const navigate = useNavigate();
  const [currentFindings, setCurrentFindings] = useState(null);
  const [reportId, setReportId] = useState(null);

  const handleUploadComplete = (findings, rId) => {
    setCurrentFindings(findings);
    setReportId(rId);
    navigate('/dashboard');
  };

  const handleNewUpload = () => {
    setCurrentFindings(null);
    setReportId(null);
    navigate('/');
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div 
            className="flex items-center gap-2 cursor-pointer"
            onClick={() => handleNewUpload()}
          >
            <div className="w-8 h-8 bg-clinical-600 rounded-lg flex items-center justify-center text-white font-bold text-xl leading-none">
              C
            </div>
            <span className="text-xl font-bold text-slate-800 tracking-tight">Clear<span className="text-clinical-600">Script</span></span>
          </div>
          
          <nav className="flex items-center gap-6">
            <a href="#" className="text-sm font-medium text-slate-600 hover:text-clinical-600 transition-colors">How it Works</a>
            <a href="#" className="text-sm font-medium text-slate-600 hover:text-clinical-600 transition-colors">Privacy</a>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 px-6 pb-12">
        <Routes>
          <Route 
            path="/" 
            element={<UploadPage onUploadComplete={handleUploadComplete} />} 
          />
          <Route 
            path="/dashboard" 
            element={
              currentFindings ? (
                <Dashboard 
                  currentFindings={currentFindings} 
                  reportId={reportId} 
                  onNewUpload={handleNewUpload}
                />
              ) : (
                <div className="flex justify-center items-center h-64 flex-col">
                  <p className="text-slate-500 mb-4">No active report session found.</p>
                  <button onClick={() => navigate('/')} className="bg-clinical-600 text-white px-6 py-2 rounded-lg font-medium hover:bg-clinical-700 transition-colors">
                    Upload a Report
                  </button>
                </div>
              )
            } 
          />
        </Routes>
      </main>
      
      {/* Footer Disclaimer */}
      <footer className="bg-slate-900 text-slate-400 py-6 mt-auto">
        <div className="max-w-7xl mx-auto px-6 text-center text-xs space-y-2">
          <p className="font-semibold text-slate-300">
            MEDICAL DISCLAIMER: CLEARSCRIPT IS NOT A MEDICAL DIAGNOSTIC TOOL.
          </p>
          <p className="max-w-4xl mx-auto">
            The AI-generated explanations provided by this application are for informational and educational purposes only. 
            They are not intended to be a substitute for professional medical advice, diagnosis, or treatment. 
            Always seek the advice of your physician or other qualified health provider with any questions you may have 
            regarding a medical condition or laboratory results.
          </p>
          <p className="mt-4 opacity-50">&copy; {new Date().getFullYear()} ClearScript AI. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppContent />
    </BrowserRouter>
  );
}
