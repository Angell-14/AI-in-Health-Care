"use client"
import { useState, useEffect } from 'react';
import { PlaceholdersAndVanishInput } from '@/components/ui/placeholders-and-vanish-input';
import { supportedLanguages } from '@/utils/languages';

const ChatInterface = () => {
  const [userInput, setUserInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedLanguage, setSelectedLanguage] = useState('english');
  const [selectedPrediction, setSelectedPrediction] = useState('chest-predict');
  const [selectedFile, setSelectedFile] = useState(null);
  const [errorMessage, setErrorMessage] = useState('');

  const placeholders = [
    "What are your symptoms?",
    "How can I assist with your health concerns?",
    // ... rest of the placeholders ...
  ];

  const predictionOptions = [
    { value: 'chest-predict', label: 'Chest X-Ray Analysis' },
    { value: 'pneumonia-predict', label: 'Pneumonia Detection' },
    { value: 'edema-predict', label: 'Edema Analysis' },
    { value: 'brain-predict', label: 'Brain Tumor Analysis' } // New dropdown option
  ];

  const handleInputChange = (e) => {
    setUserInput(e.target.value);
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!userInput.trim()) return;

    const newMessage = { type: 'user', content: userInput };
    setMessages([...messages, newMessage]);
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:5000/api/hit', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          userQuery: true,
          personalizedPlan: false,
          queryData: userInput,
          userId: 'req.user.id',
          language: selectedLanguage
        })
      });

      const data = await response.json();
      
      if (data.success && data.reply) {
        setMessages(prev => [...prev, { type: 'ai', content: data.reply }]);
      } else {
        setMessages(prev => [...prev, { type: 'ai', content: 'Sorry, I encountered an error. Please try again.' }]);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => [...prev, { type: 'ai', content: 'Sorry, I encountered an error. Please try again.' }]);
    } finally {
      setIsLoading(false);
    }

    setUserInput('');
  };

  const handleFileUpload = async (e) => {
    e.preventDefault();
    if (!selectedFile) return;

    setIsLoading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    // Determine endpoint URL based on prediction option
    let endpointUrl = '';
    if (selectedPrediction === 'brain-predict') {
      endpointUrl = 'http://localhost:8000/predictbrain';
    } else {
      endpointUrl = `http://localhost:5000/diseases/${selectedPrediction}`;
    }

    try {
      const response = await fetch(endpointUrl, {
        method: 'POST',
        // Note: For brain prediction we assume no JSON header is needed as we use FormData.
        headers: selectedPrediction !== 'brain-predict' ? {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        } : {},
        body: formData
      });

      const data = await response.json();
      
      // Create an image preview message
      const imageMessage = (
        <div>
          <p>Uploaded image for {predictionOptions.find(opt => opt.value === selectedPrediction)?.label}:</p>
          <img 
            src={URL.createObjectURL(selectedFile)} 
            alt="Uploaded" 
            className="max-w-full h-auto max-h-60 object-contain"
          />
        </div>
      );

      // For brain prediction, adjust the message accordingly
      if (selectedPrediction === 'brain-predict') {
        if (data.prediction !== undefined) {
          const resultMessage = `Brain Tumor Analysis: ${data.prediction} (Confidence: ${data.confidence}).`;
          setMessages(prev => [
            ...prev,
            { type: 'user', content: imageMessage },
            { type: 'ai', content: resultMessage }
          ]);
        } else {
          setMessages(prev => [
            ...prev,
            { type: 'ai', content: 'Sorry, brain prediction failed. Please try again.' }
          ]);
        }
      } else {
        // For other prediction options
        if (data.success) {
          const predictionLabel = predictionOptions.find(opt => opt.value === selectedPrediction)?.label;
          const resultMessage = data.prediction === 0 
            ? `Based on the analysis, there are no signs of ${predictionLabel} in the uploaded image.`
            : `The analysis indicates potential signs of ${predictionLabel}. We recommend consulting with your nearest healthcare provider for a proper medical evaluation.`;
  
          setMessages(prev => [
            ...prev,
            { type: 'user', content: imageMessage },
            { type: 'ai', content: resultMessage }
          ]);
        } else {
          setMessages(prev => [
            ...prev,
            { type: 'ai', content: 'Sorry, prediction failed. Please try again.' }
          ]);
        }
      }
    } catch (error) {
      console.error('Error uploading image:', error);
      setMessages(prev => [
        ...prev,
        { type: 'ai', content: 'Sorry, an error occurred during prediction.' }
      ]);
    } finally {
      setIsLoading(false);
      setSelectedFile(null);
      e.target.reset();
    }
  };

  const handleNearbySearch = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(async (position) => {
        const { latitude, longitude } = position.coords;
        const sessionToken = crypto.randomUUID(); // Generate a unique session token
  
        const url = 'https://api.mapbox.com/search/searchbox/v1/suggest';
        const params = new URLSearchParams({
          q: 'hospital', // Change query if needed (e.g., "doctor")
          language: 'en',
          limit: '5', // Adjust limit as desired
          session_token: sessionToken,
          proximity: `${longitude},${latitude}`, // Mapbox expects longitude,latitude order
          country: 'US',
          access_token: process.env.NEXT_PUBLIC_MAPBOX_ACCESS_TOKEN
        });
  
        try {
          const response = await fetch(`${url}?${params.toString()}`, { method: 'GET' });
          const data = await response.json();
  
          if (!response.ok) {
            setErrorMessage(data.error || 'Failed to fetch data');
            return;
          }
  
          // Build a list of suggestion items
          if (data.suggestions && data.suggestions.length > 0) {
            const suggestionsList = data.suggestions.map((suggestion, index) => (
              <li key={index}>
                {console.log(suggestion)}
                {suggestion.name || suggestion.text || 'Unknown Location'}
              </li>
            ));
            const suggestionsMessage = (
              <div>
                <h3>Nearby Hospitals/Doctors</h3>
                <ul>{suggestionsList}</ul>
                <small>{data.ttribution || data.attribution}</small>
              </div>
            );
  
            // Append the suggestions to the messages so they display in the central area
            setMessages((prev) => [...prev, { type: 'ai', content: suggestionsMessage }]);
          } else {
            setMessages((prev) => [
              ...prev,
              { type: 'ai', content: 'No nearby hospitals/doctors found.' }
            ]);
          }
          setErrorMessage('');
        } catch (error) {
          console.error('Error fetching data:', error);
          setErrorMessage('An error occurred while fetching data. Please try again.');
        }
      }, (error) => {
        console.error('Geolocation error:', error);
        setErrorMessage('Unable to retrieve your location. Please enable location services.');
      });
    } else {
      console.error('Geolocation is not supported by this browser.');
      setErrorMessage('Geolocation is not supported by this browser.');
    }
  };
  
  

  return (
    <div className="flex-1 flex flex-col h-screen relative">
      <div className="sticky top-0 z-20 w-full bg-white/80 backdrop-blur-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-2">
          <div className="flex justify-between items-center">
            <div className="flex gap-4 items-center">
              <select
                value={selectedPrediction}
                onChange={(e) => setSelectedPrediction(e.target.value)}
                className="p-2 border rounded-md bg-black text-white shadow-sm hover:bg-gray-800 transition-colors cursor-pointer"
              >
                {predictionOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              <form onSubmit={handleFileUpload} className="flex gap-2">
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e) => setSelectedFile(e.target.files[0])}
                  className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-black file:text-white hover:file:bg-gray-800 file:transition-colors file:cursor-pointer"
                />
                <button
                  type="submit"
                  disabled={!selectedFile}
                  className="px-4 py-2 bg-black text-white rounded-md hover:bg-gray-800 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Upload
                </button>
              </form>
            </div>
            <select
              value={selectedLanguage}
              onChange={(e) => setSelectedLanguage(e.target.value)}
              className="p-2 border rounded-md bg-black text-white shadow-sm w-36 hover:bg-gray-800 transition-colors cursor-pointer"
            >
              {supportedLanguages.map((lang) => (
                <option key={lang.value} value={lang.value}>
                  {lang.label}
                </option>
              ))}
            </select>
            <button
              type="button"
              onClick={handleNearbySearch}
              className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors"
            >
              Nearby Hospitals/Doctors
            </button>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto z-10 max-w-7xl mx-auto w-full p-4 space-y-4 scrollbar-hidden">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${
              message.type === 'user' ? 'justify-end' : 'justify-start'
            }`}
          >
            <div
              className={`max-w-[70%] rounded-lg p-4 ${
                message.type === 'user'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-200 text-gray-800'
              }`}
            >
              {message.content}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="max-w-[70%] rounded-lg p-4 bg-gray-200 text-gray-800">
              <div className="flex space-x-2">
                <div className="w-2 h-2 rounded-full bg-gray-600 animate-[bounce_1s_infinite_0ms]"></div>
                <div className="w-2 h-2 rounded-full bg-gray-600 animate-[bounce_1s_infinite_200ms]"></div>
                <div className="w-2 h-2 rounded-full bg-gray-600 animate-[bounce_1s_infinite_400ms]"></div>
              </div>
            </div>
          </div>
        )}
        {errorMessage && (
          <div className="p-3 bg-red-100 text-red-700 rounded mb-4">
            {errorMessage}
          </div>
        )}
      </div>

      <div className="sticky bottom-0 z-20 w-full bg-white/80 backdrop-blur-sm border-t">
        <div className="max-w-2xl mx-auto p-4">
          <div className="flex">
            <PlaceholdersAndVanishInput
              value={userInput}
              onChange={handleInputChange}
              onSubmit={handleSendMessage}
              placeholders={placeholders}
              className="flex-1 min-w-0 rounded-lg border border-gray-300 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
