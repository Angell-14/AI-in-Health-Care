import { TranslationServiceClient } from '@google-cloud/translate';

// Ensure Google Cloud credentials are set up
const projectId = process.env.GOOGLE_CLOUD_PROJECT_ID; // Add to .env file
const keyFilename = process.env.GOOGLE_APPLICATION_CREDENTIALS; // Add to .env file

const translate = new TranslationServiceClient({
  projectId,
  keyFilename,
});

// Helper function to clean text
function cleanText(text) {
  // Remove unnecessary quotes and trim whitespace
  return text.replace(/^["']|["']$/g, '').trim();
}

// Function to translate text to English
export async function convertToEnglish(text) {
  try {
    const cleanedText = cleanText(text);
    console.log("Converting to English:", cleanedText);
    const request = {
      parent: `projects/${projectId}/locations/global`,
      contents: [cleanedText],
      mimeType: "text/plain",
      sourceLanguageCode: "auto",
      targetLanguageCode: "en",
    };
    const [response] = await translate.translateText(request);
    console.log("Translation response:", response);
    if (!response || !response.translations || response.translations.length === 0) {
      throw new Error('Translation failed - empty response');
    }
    return response.translations[0].translatedText;
  } catch (error) {
    console.error("Translation to English error:", error);
    throw error;
  }
}

// Function to translate text back to the original language
export async function convertBackToUserLanguage(text, targetLanguage) {
  try {
    const cleanedText = cleanText(text);
    console.log("Converting back to:", targetLanguage, "Text:", cleanedText);
    const request = {
      parent: `projects/${projectId}/locations/global`,
      contents: [cleanedText],
      mimeType: "text/plain",
      sourceLanguageCode: "en",
      targetLanguageCode: targetLanguage,
    };
    const [response] = await translate.translateText(request);
    console.log("Translation response:", response);
    if (!response || !response.translations || response.translations.length === 0) {
      throw new Error('Translation failed - empty response');
    }
    return response.translations[0].translatedText;
  } catch (error) {
    console.error("Translation to user language error:", error);
    throw error;
  }
}

// Function to detect the original language of a given text
export async function detectLanguage(text) {
  try {
    const cleanedText = cleanText(text);
    console.log('Detecting language for cleaned text:', cleanedText);
    
    const request = {
      parent: `projects/${projectId}/locations/global`,
      content: cleanedText,
      mimeType: 'text/plain', // Added MIME type for better detection
    };
    
    const [detection] = await translate.detectLanguage(request);
    console.log('Detection result:', detection);
    
    if (!detection || !detection.languages || detection.languages.length === 0) {
      console.error('No language detected');
      return 'en'; // Default to English if detection fails
    }
    
    const detectedLanguage = detection.languages[0].languageCode;
    console.log('Detected language code:', detectedLanguage);
    return detectedLanguage;
  } catch (error) {
    console.error('Language detection error:', error);
    throw error;
  }
}