import { NextRequest, NextResponse } from "next/server";
import { readFile, stat } from "fs/promises";
import path from "path";

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

    const maxAge = 24 * 60 * 60;
    if (Date.now() - fileStat.mtimeMs > maxAge * 1000) {
      return new NextResponse("Not Found", { status: 404 });
    }

    const data = await readFile(filePath);
    return new NextResponse(data, {
      headers: {
        "Content-Type": "audio/mpeg",
        "Content-Length": data.length.toString(),
        "Cache-Control": "public, max-age=86400",
      },
    });
  } catch {
    return new NextResponse("Not Found", { status: 404 });
  }
}
