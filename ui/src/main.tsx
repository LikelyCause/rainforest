import React from "react";
import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-sans/600.css";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { setRefreshLockHandler } from "./api";
import { refetchDatasetsOnLock } from "./hooks";
import { initSaveQueue } from "./saveQueue";
import { applyThemePref } from "./theme";
import "./styles.css";
import "./styles/home.css";
import "./styles/fields.css";
import "./styles/baseline.css";
import "./styles/experiments.css";
import "./styles/tools.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 30_000, refetchOnWindowFocus: false, retry: 1 },
  },
});

initSaveQueue(queryClient);
setRefreshLockHandler(refetchDatasetsOnLock(queryClient));
applyThemePref();

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>,
);
