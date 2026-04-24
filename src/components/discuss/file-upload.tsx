"use client";

import { useCallback, useState } from "react";
import { Card } from "@/components/ui/card";

interface FileUploadProps {
  files: File[];
  onFilesChange: (files: File[]) => void;
}

const ACCEPTED_TYPES = [
  "image/jpeg",
  "image/png",
  "application/pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
];

const ACCEPTED_EXTENSIONS = ".jpg,.jpeg,.png,.pdf,.docx";

export function FileUpload({ files, onFilesChange }: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const droppedFiles = Array.from(e.dataTransfer.files).filter((f) => {
        const ext = f.name.split(".").pop()?.toLowerCase();
        return (
          ACCEPTED_TYPES.includes(f.type) ||
          (ext && [".jpg", ".jpeg", ".png", ".pdf", ".docx"].includes(`.${ext}`))
        );
      });
      onFilesChange([...files, ...droppedFiles]);
    },
    [files, onFilesChange]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files) {
        onFilesChange([...files, ...Array.from(e.target.files)]);
      }
    },
    [files, onFilesChange]
  );

  const removeFile = useCallback(
    (index: number) => {
      onFilesChange(files.filter((_, i) => i !== index));
    },
    [files, onFilesChange]
  );

  return (
    <div className="space-y-3">
      <Card
        className={`border-dashed border-2 p-8 text-center cursor-pointer transition-colors ${
          isDragging
            ? "border-primary bg-primary/5"
            : "border-muted-foreground/25 hover:border-primary/50"
        }`}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => document.getElementById("file-input")?.click()}
      >
        <div className="space-y-2">
          <div className="text-4xl">📸</div>
          <p className="text-sm text-muted-foreground">
            Drag & drop files here, or click to browse
          </p>
          <p className="text-xs text-muted-foreground">
            Supports PDF, DOCX, JPG, JPEG, PNG
          </p>
        </div>
      </Card>

      <input
        id="file-input"
        type="file"
        multiple
        accept={ACCEPTED_EXTENSIONS}
        onChange={handleChange}
        className="hidden"
      />

      {files.length > 0 && (
        <div className="space-y-2">
          {files.map((file, index) => (
            <div
              key={`${file.name}-${index}`}
              className="flex items-center justify-between rounded-md border px-3 py-2 text-sm"
            >
              <span className="truncate">{file.name}</span>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  removeFile(index);
                }}
                className="ml-2 text-muted-foreground hover:text-destructive"
              >
                x
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
