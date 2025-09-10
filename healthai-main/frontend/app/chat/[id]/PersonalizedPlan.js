"use client"
import { useState } from 'react';
import { supportedLanguages } from '@/utils/languages';

const PersonalizedPlan = () => {
  const [showQuestionnaire, setShowQuestionnaire] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [personalizedAnswers, setPersonalizedAnswers] = useState({
    question1: '',
    question2: '',
    question3: '',
    question4: '',
    question5: ''
  });
  const [personalizedPlanResponse, setPersonalizedPlanResponse] = useState('');
  const [planType, setPlanType] = useState('health');
  const [planDuration, setPlanDuration] = useState('1');
  const [error, setError] = useState('');
  const [selectedLanguage, setSelectedLanguage] = useState('english');

  const resetForm = () => {
    setShowQuestionnaire(true);
    setError('');
    setPersonalizedAnswers({
      question1: '',
      question2: '',
      question3: '',
      question4: '',
      question5: ''
    });
    setPlanType('health');
    setPlanDuration('1');
  };

  const handlePersonalizedSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    
    const formattedQueryData = `
Health Concerns: ${personalizedAnswers.question1}
Lifestyle Habits: ${personalizedAnswers.question2}
Health Goals: ${personalizedAnswers.question3}
Dietary Restrictions: ${personalizedAnswers.question4}
Medical Conditions: ${personalizedAnswers.question5}
Plan Type: ${planType}
Duration: ${planDuration} months
    `.trim();

    try {
      const response = await fetch('http://localhost:5000/api/hit', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          userQuery: false,
          personalizedPlan: true,
          queryData: formattedQueryData,
          planType,
          planDuration,
          language: selectedLanguage
        })
      });

      const data = await response.json();
      if (data.success) {
        setPersonalizedPlanResponse(data.reply);
        setShowQuestionnaire(false);
      } else {
        setError('Failed to generate plan. Please try again.');
      }
    } catch (err) {
      setError('An error occurred. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col relative min-h-[calc(100vh-4rem)]">
      <div className="max-w-4xl mx-auto w-full p-6 pb-24">
        {showQuestionnaire ? (
          <div className="bg-white rounded-lg shadow p-6">
            <div className="mb-6">
              <h2 className="text-2xl font-semibold mb-4">Personalized Plan Questionnaire</h2>
              <div className="flex items-center">
                <label className="mr-2 text-sm font-medium text-gray-700">Select Language:</label>
                <select
                  value={selectedLanguage}
                  onChange={(e) => setSelectedLanguage(e.target.value)}
                  className="p-2 border rounded-md w-36"
                >
                  {supportedLanguages.map((lang) => (
                    <option key={lang.value} value={lang.value}>
                      {lang.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            
            {error && (
              <div className="mb-4 p-3 bg-red-100 text-red-700 rounded">
                {error}
              </div>
            )}

            <form onSubmit={handlePersonalizedSubmit} className="space-y-6">
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    What are your main health concerns?
                  </label>
                  <textarea
                    value={personalizedAnswers.question1}
                    onChange={(e) => setPersonalizedAnswers(prev => ({...prev, question1: e.target.value}))}
                    className="w-full p-2 border rounded-md"
                    rows="3"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    What are your current lifestyle habits?
                  </label>
                  <textarea
                    value={personalizedAnswers.question2}
                    onChange={(e) => setPersonalizedAnswers(prev => ({...prev, question2: e.target.value}))}
                    className="w-full p-2 border rounded-md"
                    rows="3"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    What are your health goals?
                  </label>
                  <textarea
                    value={personalizedAnswers.question3}
                    onChange={(e) => setPersonalizedAnswers(prev => ({...prev, question3: e.target.value}))}
                    className="w-full p-2 border rounded-md"
                    rows="3"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Any specific dietary restrictions or preferences?
                  </label>
                  <textarea
                    value={personalizedAnswers.question4}
                    onChange={(e) => setPersonalizedAnswers(prev => ({...prev, question4: e.target.value}))}
                    className="w-full p-2 border rounded-md"
                    rows="3"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Any medical conditions or medications?
                  </label>
                  <textarea
                    value={personalizedAnswers.question5}
                    onChange={(e) => setPersonalizedAnswers(prev => ({...prev, question5: e.target.value}))}
                    className="w-full p-2 border rounded-md"
                    rows="3"
                  />
                </div>

                <div className="flex gap-4">
                  <div className="flex-1">
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Plan Type
                    </label>
                    <select
                      value={planType}
                      onChange={(e) => setPlanType(e.target.value)}
                      className="w-full p-2 border rounded-md"
                    >
                      <option value="health">Health</option>
                      <option value="fitness">Fitness</option>
                      <option value="nutrition">Nutrition</option>
                    </select>
                  </div>
                  
                  <div className="flex-1">
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Duration (months)
                    </label>
                    <select
                      value={planDuration}
                      onChange={(e) => setPlanDuration(e.target.value)}
                      className="w-full p-2 border rounded-md"
                    >
                      <option value="1">1 Month</option>
                      <option value="3">3 Months</option>
                      <option value="6">6 Months</option>
                      <option value="12">12 Months</option>
                    </select>
                  </div>
                </div>
              </div>

              <button
                type="submit"
                disabled={isLoading}
                className="w-full bg-black text-white py-2 px-4 rounded-md hover:bg-gray-800 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
              >
                {isLoading ? (
                  <div className="flex items-center justify-center space-x-2">
                    <div className="w-2 h-2 rounded-full bg-white animate-[bounce_1s_infinite_0ms]"></div>
                    <div className="w-2 h-2 rounded-full bg-white animate-[bounce_1s_infinite_200ms]"></div>
                    <div className="w-2 h-2 rounded-full bg-white animate-[bounce_1s_infinite_400ms]"></div>
                  </div>
                ) : (
                  'Generate Personalized Plan'
                )}
              </button>
            </form>
          </div>
        ) : (
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-2xl font-semibold mb-6">Your Personalized Plan</h2>
            <div className="prose max-w-none">
              {personalizedPlanResponse}
            </div>
          </div>
        )}
      </div>

      {!showQuestionnaire && (
        <div className="absolute bottom-6 right-6">
          <button
            onClick={resetForm}
            className="bg-black text-white py-2 px-4 rounded-md hover:bg-gray-800 transition-colors"
          >
            Create New Plan
          </button>
        </div>
      )}
    </div>
  );
};

export default PersonalizedPlan;