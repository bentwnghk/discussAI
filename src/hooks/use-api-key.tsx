"use client";

import {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  useRef,
  type ReactNode,
} from "react";

interface ApiKeyContextType {
  apiKey: string;
  setApiKey: (value: string) => void;
}

const ApiKeyContext = createContext<ApiKeyContextType | null>(null);

export function ApiKeyProvider({ children }: { children: ReactNode }) {
  const [apiKey, setApiKeyState] = useState("");
  const saveTimeoutRef = useRef<ReturnType<typeof setTimeout>>(undefined);

  useEffect(() => {
    fetch("/api/user/api-key")
      .then((res) => res.json())
      .then((data) => {
        if (data.apiKey) setApiKeyState(data.apiKey);
      })
      .catch(() => {});
  }, []);

  const setApiKey = useCallback((value: string) => {
    setApiKeyState(value);

    if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current);
    saveTimeoutRef.current = setTimeout(() => {
      fetch("/api/user/api-key", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ apiKey: value }),
      }).catch(() => {});
    }, 500);
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
