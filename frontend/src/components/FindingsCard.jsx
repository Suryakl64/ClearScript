import React, { useState } from 'react';
import { ChevronDown, ChevronUp, AlertCircle, CheckCircle, HelpCircle } from 'lucide-react';

const FLAG_STYLES = {
  HIGH: 'bg-finding-high/10 text-finding-high border-finding-high/20',
  LOW: 'bg-finding-high/10 text-finding-high border-finding-high/20',
  ABNORMAL: 'bg-finding-high/10 text-finding-high border-finding-high/20',
  BORDERLINE: 'bg-finding-borderline/10 text-finding-borderline border-finding-borderline/20',
  NORMAL: 'bg-finding-normal/10 text-finding-normal border-finding-normal/20',
  UNKNOWN: 'bg-finding-unknown/10 text-finding-unknown border-finding-unknown/20',
};

const FLAG_ICONS = {
  HIGH: AlertCircle,
  LOW: AlertCircle,
  ABNORMAL: AlertCircle,
  BORDERLINE: HelpCircle,
  NORMAL: CheckCircle,
  UNKNOWN: HelpCircle,
};

export default function FindingsCard({ finding }) {
  const [expanded, setExpanded] = useState(false);
  
  const flag = finding.flag ? finding.flag.toUpperCase() : 'UNKNOWN';
  const flagStyle = FLAG_STYLES[flag] || FLAG_STYLES.UNKNOWN;
  const Icon = FLAG_ICONS[flag] || FLAG_ICONS.UNKNOWN;

  // Build reference range string
  let rangeStr = "N/A";
  if (finding.range_low != null && finding.range_high != null) {
    rangeStr = `${finding.range_low} - ${finding.range_high}`;
  } else if (finding.range_high != null) {
    rangeStr = `< ${finding.range_high}`;
  } else if (finding.range_low != null) {
    rangeStr = `> ${finding.range_low}`;
  }

  // Use translated explanation if available, otherwise fallback to English explanation
  const explanation = finding.explanation_translated || finding.explanation;

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden transition-all duration-200 hover:shadow-md">
      {/* Header / Main Info */}
      <div 
        className="p-5 cursor-pointer flex items-center justify-between"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-1">
            <h3 className="font-semibold text-slate-800 text-lg">
              {finding.full_name || finding.test}
            </h3>
            <span className={`px-2.5 py-1 rounded-full text-xs font-semibold flex items-center gap-1.5 border ${flagStyle}`}>
              <Icon size={14} />
              {flag}
            </span>
          </div>
          
          <div className="flex gap-6 mt-3">
            <div>
              <p className="text-xs text-slate-500 uppercase font-medium tracking-wider mb-1">Result</p>
              <p className="text-2xl font-bold text-slate-900">
                {finding.value != null ? finding.value : '--'}
                <span className="text-sm font-normal text-slate-500 ml-1">{finding.unit}</span>
              </p>
            </div>
            <div className="border-l border-slate-200 pl-6">
              <p className="text-xs text-slate-500 uppercase font-medium tracking-wider mb-1">Normal Range</p>
              <p className="text-base text-slate-700 mt-1">
                {rangeStr} <span className="text-sm text-slate-500">{finding.unit}</span>
              </p>
            </div>
          </div>
        </div>
        
        <div className="ml-4 text-slate-400">
          {expanded ? <ChevronUp size={24} /> : <ChevronDown size={24} />}
        </div>
      </div>

      {/* Expanded Explanation */}
      {expanded && (
        <div className="px-5 py-4 bg-slate-50 border-t border-slate-100">
          <h4 className="text-sm font-semibold text-slate-700 mb-2 flex items-center gap-2">
            What this means
            {finding.explanation_available === false && (
              <span className="text-[10px] bg-slate-200 text-slate-600 px-2 py-0.5 rounded-full font-normal">
                Auto-generated
              </span>
            )}
          </h4>
          <p className="text-slate-600 leading-relaxed text-sm">
            {explanation || "No explanation available for this finding."}
          </p>
        </div>
      )}
    </div>
  );
}
