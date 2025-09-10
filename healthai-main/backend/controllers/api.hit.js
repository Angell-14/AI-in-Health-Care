import axios from 'axios';
import { convertToEnglish, convertBackToUserLanguage, detectLanguage } from "./format.js";
import FormData from 'form-data';

export const apiHit = async (req, res) => {
  try {
    let { userQuery, personalizedPlan, queryData, language } = req.body;
    
    // Append the specified string to the start of queryData
    queryData = "As RoboDoc One, do not mention names of individuals, doctors, addresses, offices, companies, or any other personally identifiable information. Your responses are purely informational and do not reference specific entities, Now answer the following: " + queryData;

    // Debug log the incoming request
    console.log('Incoming request data:', {
      userQuery,
      personalizedPlan,
      queryData,
      language
    });

    // Validate that queryData is a string
    if (typeof queryData !== 'string') {
      throw new Error('queryData must be a string');
    }

    // Log the query being sent to FastAPI
    console.log('Sending to FastAPI:', queryData);

    // Call the Python FastAPI backend with query and language
    const response = await axios.post('http://localhost:8000/chat', {
      message: queryData,
      language: language || 'english' // default to english if not provided
    });

    // Log the FastAPI response
    console.log('Received from FastAPI:', response.data);

    res.status(201).json({ 
      success: true, 
      message: "Query processed",
      reply: response.data.reply
    });

  } catch (error) {
    console.error('Full error object:', error);
    console.error('Error stack:', error.stack);
    res.status(400).json({ 
      success: false, 
      error: "Query processing failed",
      details: error.message,
      stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
    });
  }
};

export const Predict = async (req, res) => {
  try {
    // Check if file exists in request
    if (!req.file) {
      throw new Error('No file uploaded');
    }

    // Create form data
    const formData = new FormData();
    formData.append('file', req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype
    });

    // Log the request attempt
    console.log('Attempting to send image to FastAPI prediction endpoint');

    // Call the Python FastAPI backend
    const response = await axios.post('http://localhost:8000/predict', formData, {
      headers: {
        ...formData.getHeaders()
      }
    });

    // Log the FastAPI response
    console.log('Received from FastAPI:', response.data);

    res.status(200).json({
      success: true,
      message: "Image processed successfully",
      prediction: response.data.prediction
    });

  } catch (error) {
    console.error('Error processing image:', error);
    res.status(400).json({
      success: false,
      error: "Image processing failed",
      details: error.message,
      stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
    });
  }
};
  
  