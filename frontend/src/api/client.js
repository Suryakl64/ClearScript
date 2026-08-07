import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Phases 1 & 2: Extract findings from PDF/Image
export const extractFindings = async (file, prefer = null) => {
  const formData = new FormData();
  formData.append('file', file);
  if (prefer) {
    formData.append('prefer', prefer);
  }

  const response = await apiClient.post('/vision/extract', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

// Phase 3: Explain findings
export const explainFindings = async (findings) => {
  const response = await apiClient.post('/explain/generate', { findings });
  return response.data;
};

// Phase 4a: Store report for Chat
export const storeReport = async (filename, reportType, findings, summary = "") => {
  const response = await apiClient.post('/chat/store', {
    filename,
    report_type: reportType,
    findings,
    summary,
  });
  return response.data;
};

// Phase 4b: Translate findings
export const translateFindings = async (findings, targetLang) => {
  const response = await apiClient.post('/translate/findings', {
    findings,
    target_lang: targetLang,
  });
  return response.data;
};

// Phase 4c: RAG Chat
export const askQuestion = async (reportId, question) => {
  const response = await apiClient.post('/chat/ask', {
    report_id: reportId,
    question,
  });
  return response.data;
};

export const getReports = async () => {
  const response = await apiClient.get('/chat/reports');
  return response.data;
};
