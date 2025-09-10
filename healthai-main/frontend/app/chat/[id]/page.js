"use client"
import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import AuthForm from './AuthForm';
import ChatInterface from './ChatInterface';
import PersonalizedPlan from './PersonalizedPlan';
import { BackgroundBeams } from '@/components/ui/background-beams';
import { use } from 'react';
const ChatPage = ({ params }) => {
  const router = useRouter();
  const unwrappedParams = use(params);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      setIsAuthenticated(true);
    }
  }, []);

  const handleLogout = () => {
    localStorage.removeItem('token');
    setIsAuthenticated(false);
  };

  const handleRedirect = () => {
    const targetPath = unwrappedParams.id === 'ch' ? '/chat/p' : '/chat/ch';
    router.push(targetPath);
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col relative">
      <BackgroundBeams className="absolute inset-0 z-0" />
      
      {!isAuthenticated ? (
        <div className="relative z-10">
          <AuthForm onAuthentication={() => setIsAuthenticated(true)} />
        </div>
      ) : (
        <>
          <div className="bg-white/80 backdrop-blur-sm shadow-sm relative z-10">
            <div className="max-w-7xl mx-auto px-4 py-3 flex justify-between items-center">
              <h1 className="text-xl font-semibold">
                {unwrappedParams.id === 'ch' ? 'Health Chat' : 'Personalized Plan'}
              </h1>
              <div className="flex items-center">
                <button
                  onClick={handleRedirect}
                  className="px-4 py-2 text-sm text-white bg-blue-500 rounded-lg hover:bg-blue-600 transition-colors mr-2"
                >
                  {unwrappedParams.id === 'ch' ? 'Go to Personalized Plan' : 'Go to Health Chat'}
                </button>
                <button
                  onClick={handleLogout}
                  className="px-4 py-2 text-sm text-white bg-black rounded-lg hover:bg-gray-800 transition-colors"
                >
                  Logout
                </button>
              </div>
            </div>
          </div>

          <div className="flex-1 relative z-10">
            {unwrappedParams.id === 'ch' ? (
              <ChatInterface />
            ) : unwrappedParams.id === 'p' ? (
              <PersonalizedPlan />
            ) : (
              <div className="flex items-center justify-center h-full">
                <div className="text-xl text-gray-600">Nothing to show here</div>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
};

export default ChatPage;