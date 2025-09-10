import multer from 'multer';
import { Predict } from '../controllers/api.hit.js';
import express from 'express';
const router = express.Router();
const upload = multer({
  storage: multer.memoryStorage()
});

router.post('/chest-predict', upload.single('file'), Predict);
router.post('/pneumonia-predict', upload.single('file'), Predict);
router.post('/edema-predict', upload.single('file'), Predict);


export default router;