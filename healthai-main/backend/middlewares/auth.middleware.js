import jwt from "jsonwebtoken";
import { prisma } from "../index.js";

export const verifyToken = async (req, res, next) => {
  try {
    const token = req.headers.authorization?.split(" ")[1];
    if (!token) return res.status(401).json({ success:false,error: "Access denied" });

    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.user = await prisma.user.findUnique({ where: { id: decoded.userId } });
    if (!req.user) return res.status(401).json({  success:false, error: "Invalid token" });

    next();
  } catch (error) {
    res.status(401).json({  success:false, error: "Authentication failed" });
  }
};
