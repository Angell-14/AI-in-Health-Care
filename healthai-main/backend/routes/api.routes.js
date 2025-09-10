import express from "express";
import { apiHit } from "../controllers/api.hit.js";
import { verifyToken } from "../middlewares/auth.middleware.js";

const router = express.Router();

router.post("/hit", verifyToken, apiHit);

export default router; 