// controllers/auth.controller.js
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import { prisma } from "../index.js";

export const register = async (req, res) => {
  try {
    const { email, password } = req.body;
    console.log(email, password);
    const hashedPassword = await bcrypt.hash(password, 10);
    const user = await prisma.user.create({
      data: { email, password: hashedPassword },
    });
    console.log(user);
    // Create token after successful registration
    const token = jwt.sign({ userId: user.id }, process.env.JWT_SECRET, { expiresIn: "1h" });
    res.status(201).json({ success: true, message: "User registered successfully", token });
  } catch (error) {
    res.status(400).json({ success: false, error: "User registration failed" });
    console.log(error);
  }
};

export const login = async (req, res) => {
  try {
    const { email, password } = req.body;
    const user = await prisma.user.findUnique({ where: { email } });
    if (!user || !(await bcrypt.compare(password, user.password))) {
      return res.status(401).json({ success: false, error: "Invalid credentials" });
    }
    const token = jwt.sign({ userId: user.id }, process.env.JWT_SECRET, { expiresIn: "1h" });
    res.json({ success: true, token });
  } catch (error) {
    res.status(500).json({ success: false, error: "Login failed" });
  }
};

// controllers/api.controller.js
export const apiHit = async (req, res) => {
  try {
    const { userQuery, personalizedPlan, queryData } = req.body;
    const query = await prisma.query.create({
      data: {
        userQuery,
        personalizedPlan,
        queryData,
        userId: req.user.id,
      },
    });
    res.status(201).json({ success: true, message: "Query processed", query });
  } catch (error) {
    res.status(400).json({ success: false, error: "Query processing failed" });
  }
};
