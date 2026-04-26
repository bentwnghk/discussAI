"use client";

import {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  type ReactNode,
} from "react";
import { useSession } from "next-auth/react";

interface ApiKeyContextType {
  apiKey: string;
  setApiKey: (value: string) => void;
}

const ApiKeyContext = createContext<ApiKeyContextType | null>(null);

export function ApiKeyProvider({ children }: { children: ReactNode }) {
  const [apiKey, setApiKeyState] = useState("");
  const { status } = useSession();

  useEffect(() => {
    if (status !== "authenticated") return;
    fetch("/api/user/api-key")
      .then((res) => res.json())
      .then((data) => {
        if (data.apiKey) setApiKeyState(data.apiKey);
      })
      .catch(() => {});
  }, [status]);

  const setApiKey = useCallback((value: string) => {
    setApiKeyState(value);
    fetch("/api/user/api-key", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ apiKey: value }),
    }).catch(() => {});
  }, []);

  return (
    <ApiKeyContext.Provider value={{ apiKey, setApiKey }}>
      {children}
    </ApiKeyContext.Provider>
  );
}

export function useApiKey() {
  const context = useContext(ApiKeyContext);
  if (!context)
    throw new Error("useApiKey must be used within ApiKeyProvider");
  return context;
}
