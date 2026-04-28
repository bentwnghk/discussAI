import { NextRequest, NextResponse } from "next/server";
import { readFile, stat } from "fs/promises";
import path from "path";
import { AUDIO_TTL_MS } from "@/lib/audio-ttl";

const AUDIO_DIR = path.join(process.cwd(), "tmp", "audio");

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  try {
    const { path: segments } = await params;
    const filename = segments.join("/");

    if (filename.includes("..")) {
      return new NextResponse("Forbidden", { status: 403 });
    }

    const filePath = path.join(AUDIO_DIR, filename);
    const fileStat = await stat(filePath);

    if (Date.now() - fileStat.mtimeMs > AUDIO_TTL_MS) {
      return new NextResponse("Not Found", { status: 404 });
    }

    const data = await readFile(filePath);
    const ttlSeconds = Math.ceil(
      (AUDIO_TTL_MS - (Date.now() - fileStat.mtimeMs)) / 1000
    );
    return new NextResponse(data, {
      headers: {
        "Content-Type": "audio/mpeg",
        "Content-Length": data.length.toString(),
        "Cache-Control": `public, max-age=${ttlSeconds}`,
      },
    });
  } catch {
    return new NextResponse("Not Found", { status: 404 });
  }
}
