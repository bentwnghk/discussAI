import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { generateDocx } from "@/lib/export/docx-generator";
import type { DialogueItem, LearningNotes } from "@/types";

export async function POST(req: NextRequest) {
  try {
    const session = await auth();
    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await req.json();
    const { transcript, learningNotes, title, extractedText, accessCode } = body as {
      transcript: DialogueItem[];
      learningNotes: LearningNotes;
      title?: string;
      extractedText?: string;
      accessCode?: string;
    };

    if (!transcript || !learningNotes) {
      return NextResponse.json(
        { error: "Transcript and learning notes are required." },
        { status: 400 }
      );
    }

    const buffer = await generateDocx(transcript, learningNotes, title, extractedText, accessCode, req.nextUrl.origin);

    const timestamp = new Date().toLocaleString("en-HK", { timeZone: "Asia/Hong_Kong" }).replace(/[/:, ]/g, "-");
    const filename = `Mr.NG-DiscussAI-notes-${timestamp}.docx`;

    return new NextResponse(new Uint8Array(buffer), {
      headers: {
        "Content-Type":
          "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "Content-Disposition": `attachment; filename="${filename}"`,
        "Content-Length": buffer.length.toString(),
      },
    });
  } catch (error) {
    console.error("DOCX export error:", error);
    return NextResponse.json(
      { error: "Failed to generate document." },
      { status: 500 }
    );
  }
}
