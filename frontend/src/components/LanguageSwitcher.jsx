import React from 'react';
import { Languages, Loader2 } from 'lucide-react';

const LANGUAGES = [
  { code: 'en', label: 'English' },
  { code: 'hi', label: 'Hindi' },
  { code: 'ta', label: 'Tamil' },
  { code: 'kn', label: 'Kannada' },
  { code: 'te', label: 'Telugu' },
];

export default function LanguageSwitcher({ currentLang, onLanguageChange, isLoading }) {
  return (
    <div className="flex items-center gap-3 bg-white px-4 py-2 rounded-lg border border-slate-200 shadow-sm">
      <Languages size={18} className="text-slate-500" />
      <span className="text-sm font-medium text-slate-700">Translate to:</span>
      
      <div className="relative">
        <select
          value={currentLang}
          onChange={(e) => onLanguageChange(e.target.value)}
          disabled={isLoading}
          className="appearance-none bg-slate-50 border border-slate-200 text-slate-700 text-sm rounded-md focus:ring-clinical-500 focus:border-clinical-500 block w-full pl-3 pr-8 py-1.5 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {LANGUAGES.map((lang) => (
            <option key={lang.code} value={lang.code}>
              {lang.label}
            </option>
          ))}
        </select>
        {isLoading && (
          <div className="absolute right-2 top-1/2 -translate-y-1/2">
            <Loader2 size={14} className="animate-spin text-clinical-600" />
          </div>
        )}
      </div>
    </div>
  );
}
