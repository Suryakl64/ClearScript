import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

export default function TrendChart({ data, title, unit, referenceLow, referenceHigh }) {
  // data should be array of objects: { date: 'Jan 1', value: 12.5 }
  
  if (!data || data.length === 0) return null;

  return (
    <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
      <h3 className="font-semibold text-slate-800 mb-4">{title} Trend</h3>
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
            <XAxis 
              dataKey="date" 
              tick={{ fontSize: 12, fill: '#64748b' }} 
              axisLine={false} 
              tickLine={false}
              dy={10}
            />
            <YAxis 
              tick={{ fontSize: 12, fill: '#64748b' }} 
              axisLine={false} 
              tickLine={false}
              dx={-10}
            />
            <Tooltip 
              contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
              itemStyle={{ color: '#0f766e', fontWeight: 600 }}
              formatter={(value) => [`${value} ${unit}`, 'Result']}
            />
            
            {/* Reference Ranges */}
            {referenceHigh !== null && (
              <ReferenceLine y={referenceHigh} stroke="#ef4444" strokeDasharray="3 3" label={{ position: 'top', value: 'High', fill: '#ef4444', fontSize: 10 }} />
            )}
            {referenceLow !== null && (
              <ReferenceLine y={referenceLow} stroke="#ef4444" strokeDasharray="3 3" label={{ position: 'bottom', value: 'Low', fill: '#ef4444', fontSize: 10 }} />
            )}

            <Line 
              type="monotone" 
              dataKey="value" 
              stroke="#0d9488" 
              strokeWidth={3}
              dot={{ r: 4, strokeWidth: 2, fill: '#fff', stroke: '#0d9488' }}
              activeDot={{ r: 6, fill: '#0d9488', stroke: '#fff', strokeWidth: 2 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
