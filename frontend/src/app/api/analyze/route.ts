import { NextRequest, NextResponse } from "next/server";
import { analyzeEssayLocal } from "@/lib/analyzer";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const text = body.text;

    if (!text || typeof text !== "string" || text.trim().length < 10) {
      return NextResponse.json(
        { error: "Essay text must be at least 10 characters." },
        { status: 400 }
      );
    }

    const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:8000";
    
    // Attempt to connect to Python FastAPI backend
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3500);

      const response = await fetch(`${BACKEND_URL}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (response.ok) {
        const data = await response.json();
        // Return full backend response, marking it as neural model active
        return NextResponse.json({
          ...data,
          is_fallback: false,
          model_badge: "Neural (GPT-2 124M) + 9-Signal Ensemble",
        });
      }
    } catch {
      // Backend not running or unreachable — fallback gracefully to built-in TypeScript engine
      console.log("[api/analyze] FastAPI backend unreachable. Executing built-in 9-signal fallback engine.");
    }

    // Run robust built-in TypeScript 9-signal analytical engine
    const localResult = analyzeEssayLocal(text);
    return NextResponse.json(localResult);
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Internal analysis error.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
