// React component for resume analysis

import React, { useState } from 'react';
import axios from 'axios';

function ResumeAnalyzer() {
  const [resume, setResume] = useState(null);
  const [jdText, setJdText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async () => {
    if (!resume || !jdText) {
      alert('Please provide both resume and job description');
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('resume', resume);
    formData.append('jd_text', jdText);

    try {
      const response = await axios.post(
        'http://localhost:8000/api/resume/analyze',
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' }
        }
      );
      setResult(response.data);
    } catch (error) {
      console.error('Analysis failed:', error);
      alert('Analysis failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const getScoreColor = (score) => {
    if (score >= 75) return 'text-green-600';
    if (score >= 50) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">Resume Intelligence Analyzer</h1>
      
      {/* Input Section */}
      <div className="mb-6">
        <label className="block mb-2 font-semibold">Upload Resume (PDF)</label>
        <input
          type="file"
          accept=".pdf"
          onChange={(e) => setResume(e.target.files[0])}
          className="border p-2 rounded w-full"
        />
      </div>

      <div className="mb-6">
        <label className="block mb-2 font-semibold">Job Description</label>
        <textarea
          value={jdText}
          onChange={(e) => setJdText(e.target.value)}
          rows={10}
          className="border p-2 rounded w-full"
          placeholder="Paste job description here..."
        />
      </div>

      <button
        onClick={handleAnalyze}
        disabled={loading}
        className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700 disabled:bg-gray-400"
      >
        {loading ? 'Analyzing...' : 'Analyze Resume'}
      </button>

      {/* Results Section */}
      {result && (
        <div className="mt-8">
          <div className="bg-white shadow-lg rounded-lg p-6 mb-6">
            <h2 className="text-2xl font-bold mb-4">Analysis Results</h2>
            <div className="mb-4">
              <span className="text-lg font-semibold">Overall Match: </span>
              <span className={`text-2xl font-bold ${getScoreColor(result.overall_match_score)}`}>
                {result.overall_match_score}%
              </span>
            </div>
            <p className="text-gray-700">{result.summary}</p>
          </div>

          {/* Section Scores */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
            {Object.entries(result.sections).map(([section, data]) => (
              <div key={section} className="bg-white shadow rounded-lg p-4">
                <h3 className="font-bold text-lg mb-2 capitalize">
                  {section.replace('_', ' ')}
                </h3>
                <div className={`text-3xl font-bold ${getScoreColor(data.score)}`}>
                  {data.score}%
                </div>
                {data.comment && (
                  <p className="text-sm text-gray-600 mt-2">{data.comment}</p>
                )}
              </div>
            ))}
          </div>

          {/* Detailed Sections */}
          {Object.entries(result.sections).map(([section, data]) => (
            <div key={section} className="bg-white shadow-lg rounded-lg p-6 mb-6">
              <h3 className="text-xl font-bold mb-4 capitalize">
                {section.replace('_', ' ')} Details
              </h3>
              
              {/* Skills Section */}
              {section === 'skills' && (
                <>
                  <div className="mb-4">
                    <h4 className="font-semibold text-green-600 mb-2">✓ Matched Skills</h4>
                    <div className="flex flex-wrap gap-2">
                      {data.matched_skills.map((skill, idx) => (
                        <span key={idx} className="bg-green-100 text-green-800 px-3 py-1 rounded">
                          {skill}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="mb-4">
                    <h4 className="font-semibold text-red-600 mb-2">✗ Missing Skills</h4>
                    <div className="flex flex-wrap gap-2">
                      {data.missing_skills.map((skill, idx) => (
                        <span key={idx} className="bg-red-100 text-red-800 px-3 py-1 rounded">
                          {skill}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="mb-4">
                    <h4 className="font-semibold text-yellow-600 mb-2">⚠ Mentioned but Not Listed</h4>
                    <div className="flex flex-wrap gap-2">
                      {data.mentioned_but_not_listed.map((skill, idx) => (
                        <span key={idx} className="bg-yellow-100 text-yellow-800 px-3 py-1 rounded">
                          {skill}
                        </span>
                      ))}
                    </div>
                  </div>
                </>
              )}

              {/* Suggestions */}
              {data.suggestions && data.suggestions.length > 0 && (
                <div className="mt-4">
                  <h4 className="font-semibold mb-2">💡 Suggestions</h4>
                  <ul className="list-disc list-inside space-y-1">
                    {data.suggestions.map((suggestion, idx) => (
                      <li key={idx} className="text-gray-700">{suggestion}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ))}

          {/* Highlight Instructions */}
          <div className="bg-white shadow-lg rounded-lg p-6">
            <h3 className="text-xl font-bold mb-4">Highlighting Guide</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <h4 className="font-semibold text-green-600 mb-2">🟢 Keep/Emphasize</h4>
                <ul className="text-sm space-y-1">
                  {result.highlight_instructions.green.map((item, idx) => (
                    <li key={idx}>{item}</li>
                  ))}
                </ul>
              </div>
              <div>
                <h4 className="font-semibold text-yellow-600 mb-2">🟡 Improve/Add</h4>
                <ul className="text-sm space-y-1">
                  {result.highlight_instructions.yellow.map((item, idx) => (
                    <li key={idx}>{item}</li>
                  ))}
                </ul>
              </div>
              <div>
                <h4 className="font-semibold text-red-600 mb-2">🔴 Remove</h4>
                <ul className="text-sm space-y-1">
                  {result.highlight_instructions.red.map((item, idx) => (
                    <li key={idx}>{item}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default ResumeAnalyzer;
