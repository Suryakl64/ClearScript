import React, { useState, useCallback } from 'react';
import { UploadCloud, FileText, Image as ImageIcon, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';
import { extractFindings, explainFindings, storeReport } from '../api/client';

const STEPS = [
  { id: 'upload', label: 'Upload Report', loadingText: 'Uploading...' },
  { id: 'vision', label: 'Extract Data (OCR/Vision)', loadingText: 'Extracting text and tables...' },
  { id: 'ner', label: 'Analyze Findings (NER)', loadingText: 'Structuring medical data...' },
  { id: 'explain', label: 'Generate Explanations', loadingText: 'Translating medical jargon...' },
  { id: 'store', label: 'Index for Chat', loadingText: 'Preparing AI chat...' },
  { id: 'complete', label: 'Complete', loadingText: 'Done' }
];

export default function UploadPage({ onUploadComplete }) {
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState(null);
  const [error, setError] = useState(null);
  
  // Progress tracking
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setIsDragging(true);
    } else if (e.type === 'dragleave') {
      setIsDragging(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  }, []);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const processFile = async (selectedFile) => {
    setError(null);
    setFile(selectedFile);
    setIsProcessing(true);
    setCurrentStepIndex(1); // Start vision step

    try {
      // Step 1: Vision / OCR + NER
      const visionRes = await extractFindings(selectedFile);
      if (!visionRes.success) throw new Error(visionRes.detail || "Extraction failed");
      
      let findings = visionRes.findings;
      const reportType = visionRes.report_type || "structured";

      // Step 3: Explanations
      setCurrentStepIndex(3);
      if (findings && findings.length > 0) {
        const explainRes = await explainFindings(findings);
        if (explainRes.success) {
          findings = explainRes.findings;
        }
      }

      // Step 4: Store & Index
      setCurrentStepIndex(4);
      let reportId = null;
      if (findings && findings.length > 0) {
        const storeRes = await storeReport(selectedFile.name, reportType, findings);
        if (storeRes.success) {
          reportId = storeRes.report_id;
        }
      }

      // Complete
      setCurrentStepIndex(5);
      setTimeout(() => {
        setIsProcessing(false);
        onUploadComplete(findings, reportId);
      }, 1000);

    } catch (err) {
      console.error(err);
      setError(err.message || "An unexpected error occurred during processing.");
      setIsProcessing(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto mt-12">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-slate-900 mb-3">Understand Your Medical Report</h1>
        <p className="text-slate-600">Upload your lab results or discharge summary to get plain-English explanations.</p>
      </div>

      <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-8">
        {!isProcessing && currentStepIndex === 0 ? (
          // Upload Area
          <div
            className={`border-2 border-dashed rounded-xl p-12 text-center transition-colors cursor-pointer
              ${isDragging ? 'border-clinical-500 bg-clinical-50' : 'border-slate-300 hover:border-clinical-400 hover:bg-slate-50'}
            `}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => document.getElementById('file-upload').click()}
          >
            <input
              id="file-upload"
              type="file"
              className="hidden"
              accept=".pdf,image/png,image/jpeg,image/jpg,image/webp"
              onChange={handleFileChange}
            />
            <div className="flex justify-center mb-4 text-clinical-600">
              <UploadCloud size={48} strokeWidth={1.5} />
            </div>
            <h3 className="text-lg font-semibold text-slate-800 mb-1">Drag and drop your report</h3>
            <p className="text-sm text-slate-500 mb-6">Supports PDF, JPG, PNG (Max 20MB)</p>
            <button className="bg-clinical-600 hover:bg-clinical-700 text-white px-6 py-2.5 rounded-lg font-medium transition-colors shadow-sm">
              Browse Files
            </button>
          </div>
        ) : (
          // Processing Area
          <div className="py-6">
            <div className="flex items-center gap-4 mb-8 p-4 bg-slate-50 rounded-lg border border-slate-100">
              <div className="w-12 h-12 bg-white rounded-lg shadow-sm border border-slate-200 flex items-center justify-center text-clinical-600">
                {file?.type?.includes('pdf') ? <FileText size={24} /> : <ImageIcon size={24} />}
              </div>
              <div className="flex-1 min-w-0">
                <p className="font-semibold text-slate-800 truncate">{file?.name}</p>
                <p className="text-xs text-slate-500">{(file?.size / 1024 / 1024).toFixed(2)} MB</p>
              </div>
            </div>

            <div className="space-y-6 px-2">
              {STEPS.map((step, idx) => {
                const isPast = idx < currentStepIndex;
                const isCurrent = idx === currentStepIndex;
                const isFuture = idx > currentStepIndex;
                
                // Skip 'upload' step in this view
                if (idx === 0) return null;

                return (
                  <div key={step.id} className={`flex items-start gap-4 ${isFuture ? 'opacity-40' : 'opacity-100'}`}>
                    <div className="mt-0.5">
                      {isPast ? (
                        <CheckCircle size={20} className="text-clinical-500" />
                      ) : isCurrent ? (
                        <Loader2 size={20} className="text-clinical-600 animate-spin" />
                      ) : (
                        <div className="w-5 h-5 rounded-full border-2 border-slate-300" />
                      )}
                    </div>
                    <div>
                      <p className={`font-medium ${isCurrent ? 'text-clinical-700' : 'text-slate-700'}`}>
                        {step.label}
                      </p>
                      {isCurrent && (
                        <motion.p 
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          className="text-sm text-slate-500 mt-1"
                        >
                          {step.loadingText}
                        </motion.p>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
            
            {error && (
              <div className="mt-8 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
                <AlertCircle className="text-red-600 mt-0.5" size={20} />
                <div>
                  <h4 className="font-semibold text-red-800">Processing Failed</h4>
                  <p className="text-sm text-red-600 mt-1">{error}</p>
                  <button 
                    onClick={() => {
                      setIsProcessing(false);
                      setCurrentStepIndex(0);
                      setError(null);
                      setFile(null);
                    }}
                    className="mt-3 text-sm font-medium text-red-700 hover:text-red-800 underline"
                  >
                    Try Again
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Trust Badges */}
      <div className="flex justify-center gap-8 mt-12 text-slate-400">
        <div className="flex items-center gap-2 text-sm">
          <CheckCircle size={16} /> <span>100% Private</span>
        </div>
        <div className="flex items-center gap-2 text-sm">
          <CheckCircle size={16} /> <span>Local AI Models</span>
        </div>
        <div className="flex items-center gap-2 text-sm">
          <CheckCircle size={16} /> <span>No Data Stored Online</span>
        </div>
      </div>
    </div>
  );
}
