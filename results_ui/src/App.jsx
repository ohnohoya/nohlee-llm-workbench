import React, { useEffect, useMemo, useState } from "react";
import { marked } from "marked";

const OUTPUT_DIR = "output";

function safeJsonParse(text) {
  try {
    return { data: JSON.parse(text), error: null };
  } catch (err) {
    return { data: null, error: err?.message || String(err) };
  }
}

function Group({ title, open, onToggle, children }) {
  return (
    <div className="group">
      <div className="group-header" onClick={onToggle}>
        <span className="group-title">{title}</span>
        <span>{open ? "▾" : "▸"}</span>
      </div>
      {open ? <div className="group-body">{children}</div> : null}
    </div>
  );
}

export default function App() {
  const [files, setFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState("");
  const [results, setResults] = useState([]);
  const [status, setStatus] = useState("No file loaded yet.");
  const [error, setError] = useState(null);
  const [serverAvailable, setServerAvailable] = useState(true);
  const [forcedOpen, setForcedOpen] = useState(null);
  const [openMap, setOpenMap] = useState({});
  const [compactMode, setCompactMode] = useState(true);
  const [order, setOrder] = useState([]);
  const [dragIndex, setDragIndex] = useState(null);
  const [dragOverIndex, setDragOverIndex] = useState(null);
  const [runConfig, setRunConfig] = useState({
    modelNames: ["gpt-4o"],
    maxTokens: "1000",
    temperatures: ["auto"],
    apiTypes: ["responses"],
    reasoningEfforts: ["auto"],
    applyParamCorrections: true,
    requests: [
      {
        label: "request_1",
        prompt: "Write a short summary of the benefits of clean code.",
      },
    ],
  });
  const [temperatureInput, setTemperatureInput] = useState("");
  const [runBusy, setRunBusy] = useState(false);
  const [runPanelOpen, setRunPanelOpen] = useState(false);
  const [modelOptions, setModelOptions] = useState({
    models: [],
    apiTypes: ["responses", "chat_completions"],
    reasoningEfforts: [],
  });

  const normalized = useMemo(() => {
    if (!Array.isArray(results)) return [];
    return results.map((entry, index) => ({ entry, index }));
  }, [results]);

  useEffect(() => {
    setOrder(normalized.map((item) => item.index));
  }, [normalized]);

  const setAllOpen = (value) => {
    setForcedOpen(value);
  };

  const handleToggle = (cardIndex, groupKey) => {
    setForcedOpen(null);
    setOpenMap((prev) => {
      const next = { ...prev };
      const cardState = { ...(next[cardIndex] || {}) };
      cardState[groupKey] = !cardState[groupKey];
      next[cardIndex] = cardState;
      return next;
    });
  };

  const fetchFiles = async () => {
    try {
      const res = await fetch("/files", { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const payload = await res.json();
      const list = payload?.files ?? [];
      setFiles(list);
      setServerAvailable(true);
      if (list.length && !selectedFile) {
        setSelectedFile(list[0].name);
      }
    } catch (err) {
      setFiles([]);
      setServerAvailable(false);
      setError(err?.message || String(err));
      setStatus("Server unavailable. You can still upload a JSON file.");
    }
  };

  useEffect(() => {
    fetchFiles();
  }, []);

  useEffect(() => {
    if (!serverAvailable) return undefined;
    const timer = window.setInterval(() => {
      fetchFiles();
    }, 5000);
    return () => window.clearInterval(timer);
  }, [serverAvailable]);

  useEffect(() => {
    if (selectedFile && serverAvailable) {
      handleLoadPath();
    }
  }, [selectedFile, serverAvailable]);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const res = await fetch("/models", { cache: "no-store" });
        if (!res.ok) return;
        const payload = await res.json();
        setModelOptions({
          models: payload?.models ?? [],
          apiTypes: payload?.api_types ?? ["responses", "chat_completions"],
          reasoningEfforts: payload?.reasoning_efforts ?? [],
        });
      } catch {
        // ignore; fallback to defaults
      }
    };
    fetchModels();
  }, []);

  const handleFileUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const text = await file.text();
    const { data, error: parseError } = safeJsonParse(text);
    if (parseError) {
      setError(parseError);
      setStatus("Failed to parse JSON.");
      return;
    }
    setResults(data);
    setError(null);
    setStatus(`Loaded ${Array.isArray(data) ? data.length : 0} records from ${file.name}.`);
    setForcedOpen("all");
  };

  const handleLoadPath = async () => {
    try {
      const target = selectedFile ? `${OUTPUT_DIR}/${selectedFile}` : "";
      if (!target) {
        setStatus("Select a file from the dropdown.");
        return;
      }
      const res = await fetch(target, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const text = await res.text();
      const { data, error: parseError } = safeJsonParse(text);
      if (parseError) throw new Error(parseError);
      setResults(data);
      setError(null);
      setStatus(`Loaded ${Array.isArray(data) ? data.length : 0} records from ${target}.`);
      setForcedOpen("all");
    } catch (err) {
      setError(err?.message || String(err));
      setStatus("Failed to load from path.");
    }
  };

  const getSelectValues = (event) =>
    Array.from(event.target.selectedOptions).map((opt) => opt.value);

  const normalizeTemperatureToken = (rawValue) => {
    const value = rawValue.trim();
    if (!value) {
      throw new Error("Temperature cannot be empty.");
    }
    const lower = value.toLowerCase();
    if (lower === "auto" || lower === "null") {
      return "auto";
    }
    const num = Number(value);
    if (!Number.isFinite(num)) {
      throw new Error(`Invalid temperature value: ${value}`);
    }
    return String(num);
  };

  const addTemperatureTokens = () => {
    if (!temperatureInput.trim()) return;
    try {
      const parsedTokens = temperatureInput
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean)
        .map(normalizeTemperatureToken);

      setRunConfig((prev) => {
        const seen = new Set(prev.temperatures);
        const next = [...prev.temperatures];
        parsedTokens.forEach((token) => {
          if (!seen.has(token)) {
            next.push(token);
            seen.add(token);
          }
        });
        return { ...prev, temperatures: next };
      });
      setTemperatureInput("");
      setError(null);
    } catch (err) {
      setError(err?.message || String(err));
      setStatus("Invalid temperature input.");
    }
  };

  const removeTemperatureToken = (tokenToRemove) => {
    setRunConfig((prev) => ({
      ...prev,
      temperatures: prev.temperatures.filter((token) => token !== tokenToRemove),
    }));
  };

  const handleRunSubmit = async () => {
    if (runBusy) return;
    setError(null);
    try {
      const modelNames = runConfig.modelNames;
      const apiTypes = runConfig.apiTypes;
      const temperatures = runConfig.temperatures.map((value) => {
        if (value === "auto") return null;
        const num = Number(value);
        if (!Number.isFinite(num)) {
          throw new Error(`Invalid temperature value: ${value}`);
        }
        return num;
      });
      if (!temperatures.length) {
        throw new Error("Provide at least one temperature value.");
      }
      const reasoningEfforts = runConfig.reasoningEfforts;
      const maxTokens = Number(runConfig.maxTokens);
      if (!Number.isFinite(maxTokens) || maxTokens <= 0) {
        throw new Error("Max tokens must be a positive number.");
      }
      const requests = runConfig.requests
        .map((item) => ({
          label: item.label?.trim() || null,
          prompt: item.prompt?.trim() || null,
        }))
        .filter((item) => item.prompt);
      if (!requests.length) {
        throw new Error("Provide at least one prompt.");
      }

      const payload = {
        model_names: modelNames,
        max_tokens: Math.floor(maxTokens),
        temperatures,
        api_types: apiTypes,
        reasoning_efforts: reasoningEfforts,
        apply_param_corrections: runConfig.applyParamCorrections,
        requests,
        return_results: false,
      };

      setRunBusy(true);
      setStatus("Submitting run...");
      const res = await fetch("/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `HTTP ${res.status}`);
      }
      const result = await res.json();
      setStatus(`Run submitted. Wrote ${result?.count ?? 0} records.`);
      await fetchFiles();
    } catch (err) {
      setError(err?.message || String(err));
      setStatus("Failed to submit run.");
    } finally {
      setRunBusy(false);
    }
  };

  const renderValue = (value) => {
    if (value === null) {
      return <pre>null</pre>;
    }
    if (typeof value === "string") {
      return <pre>{value}</pre>;
    }
    return <pre>{JSON.stringify(value ?? {}, null, 2)}</pre>;
  };

  const renderMarkdownOutput = (value) => {
    const renderMarkdownBlock = (text) => (
      <div className="markdown" dangerouslySetInnerHTML={{ __html: marked.parse(text || "") }} />
    );

    if (value && typeof value === "object") {
      return (
        <div className="response-grid">
          {Object.entries(value).map(([k, v]) => (
            <div key={k} className="response-field">
              <div className="response-label">{k}</div>
              {typeof v === "string" ? renderMarkdownBlock(v) : renderValue(v)}
            </div>
          ))}
        </div>
      );
    }

    if (typeof value !== "string") {
      return <pre>{value === null ? "null" : JSON.stringify(value ?? {}, null, 2)}</pre>;
    }

    let parsed = null;
    let text = value;
    const fenceMatch = text.match(/^```[a-zA-Z0-9_-]*\n([\s\S]*?)\n```$/);
    if (fenceMatch) {
      text = fenceMatch[1];
    }
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = null;
    }

    if (parsed && typeof parsed === "object") {
      return (
        <div className="response-grid">
          {Object.entries(parsed).map(([k, v]) => (
            <div key={k} className="response-field">
              <div className="response-label">{k}</div>
              {typeof v === "string" ? renderMarkdownBlock(v) : renderValue(v)}
            </div>
          ))}
        </div>
      );
    }

    return renderMarkdownBlock(text);
  };

  const renderResponse = (response) => {
    if (!response || typeof response !== "object") return renderValue(response);
    return (
      <div className="response-grid">
        {Object.entries(response).map(([key, value]) => (
          <div key={key} className="response-field">
            <div className="response-label">{key}</div>
            {key === "output"
              ? renderMarkdownOutput(value)
              : key === "metadata" && value && typeof value === "object"
                ? (
                  <div className="metadata-grid">
                    {Object.entries(value).map(([metaKey, metaValue]) => (
                      <div key={metaKey} className="metadata-row">
                        <div className="metadata-key">{metaKey}</div>
                        <div className="metadata-value">{renderValue(metaValue)}</div>
                      </div>
                    ))}
                  </div>
                )
                : renderValue(value)}
          </div>
        ))}
      </div>
    );
  };

  const renderKeyValueGrid = (value) => {
    if (!value || typeof value !== "object") return renderValue(value);
    const entries = Object.entries(value);
    const rank = (key) => {
      if (key === "sid") return 0;
      if (key === "temperature") return 1;
      if (key === "temperature_warning") return 2;
      return 3;
    };
    const orderedEntries = entries.sort(([a], [b]) => {
      const ra = rank(a);
      const rb = rank(b);
      if (ra !== rb) return ra - rb;
      return 0;
    });
    return (
      <div className="metadata-grid">
        {orderedEntries.map(([metaKey, metaValue]) => (
          <div key={metaKey} className="metadata-row">
            <div className="metadata-key">{metaKey}</div>
            <div className="metadata-value">{renderValue(metaValue)}</div>
          </div>
        ))}
      </div>
    );
  };

  const renderGroup = (cardIndex, key, data) => {
    const isOpen = forcedOpen === null ? openMap?.[cardIndex]?.[key] ?? false : forcedOpen === "all";
    return (
      <Group
        key={`${cardIndex}-${key}`}
        title={key}
        open={isOpen}
        onToggle={() => handleToggle(cardIndex, key)}
      >
        {key === "llm_response"
          ? renderResponse(data)
          : key === "llm_call_config" || key === "llm_request"
            ? renderKeyValueGrid(data)
            : renderValue(data)}
      </Group>
    );
  };

  const extractTextFromContent = (content) => {
    if (typeof content === "string") return content;
    if (!Array.isArray(content)) return "";
    return content
      .map((part) => {
        if (typeof part === "string") return part;
        if (!part || typeof part !== "object") return "";
        if (typeof part.text === "string") return part.text;
        if (typeof part.input_text === "string") return part.input_text;
        if (typeof part.content === "string") return part.content;
        return "";
      })
      .filter(Boolean)
      .join("\n");
  };

  const extractPromptFromMessages = (messages) => {
    if (!Array.isArray(messages)) return "";
    const userMessages = messages.filter((message) => message?.role === "user");
    const source = userMessages.length ? userMessages : messages;
    return source
      .map((message) => extractTextFromContent(message?.content))
      .filter(Boolean)
      .join("\n\n");
  };

  const getUserPromptText = (entry) => {
    const request = entry?.llm_request || {};
    const payload = request?.payload || {};
    const source = payload?.input ?? payload?.messages ?? request?.messages ?? request?.prompt;
    if (typeof source === "string") return source;
    if (Array.isArray(source)) return extractPromptFromMessages(source);
    return "";
  };

  const orderedModels = useMemo(() => {
    const models = modelOptions.models.length ? [...modelOptions.models] : ["gpt-4o"];
    const rankModel = (name) => {
      const lower = name.toLowerCase();
      if (lower.startsWith("gpt-")) {
        const match = lower.match(/^gpt-(\d+)(?:\.(\d+))?/);
        const major = match ? Number(match[1]) : 0;
        const minor = match && match[2] ? Number(match[2]) : 0;
        return { group: 0, major, minor, name };
      }
      if (lower.startsWith("o")) {
        const match = lower.match(/^o(\d+)/);
        const major = match ? Number(match[1]) : 0;
        return { group: 1, major, minor: 0, name };
      }
      return { group: 2, major: 0, minor: 0, name };
    };

    models.sort((a, b) => {
      const ra = rankModel(a);
      const rb = rankModel(b);
      if (ra.group !== rb.group) return ra.group - rb.group;
      if (ra.major !== rb.major) return rb.major - ra.major;
      if (ra.minor !== rb.minor) return rb.minor - ra.minor;
      return ra.name.localeCompare(rb.name);
    });

    return models;
  }, [modelOptions.models]);

  const topModels = orderedModels.slice(0, 10);
  const reorderList = (list, fromItem, toItem) => {
    const next = [...list];
    const from = next.indexOf(fromItem);
    const to = next.indexOf(toItem);
    if (from === -1 || to === -1) return list;
    next.splice(from, 1);
    next.splice(to, 0, fromItem);
    return next;
  };
  const displayOrder = useMemo(() => {
    if (dragIndex === null || dragOverIndex === null || dragIndex === dragOverIndex) {
      return order;
    }
    return reorderList(order, dragIndex, dragOverIndex);
  }, [order, dragIndex, dragOverIndex]);

  return (
    <div>
      <header>
        <h1>LLM Results Viewer</h1>
        <p>Run prompt experiments and compare outputs side by side.</p>
      </header>

      <section className={`config-panel ${runPanelOpen ? "open" : "collapsed"}`}>
        <div className="panel-header">
          <div>
            <h2>Run Setup</h2>
            <p>Choose models and prompts, then submit a batch run.</p>
          </div>
          <div className="panel-actions">
            <button
              type="button"
              className="secondary"
              onClick={() => setRunPanelOpen((prev) => !prev)}
            >
              {runPanelOpen ? "Collapse" : "Expand"}
            </button>
            <button onClick={handleRunSubmit} disabled={!serverAvailable || runBusy}>
              {runBusy ? "Submitting..." : "Submit Run"}
            </button>
          </div>
        </div>
        {runPanelOpen ? (
          <div className="config-grid">
          <div className="control-block">
            <label>Model Names</label>
            <select
              multiple
              value={runConfig.modelNames}
              onChange={(e) =>
                setRunConfig((prev) => ({ ...prev, modelNames: getSelectValues(e) }))
              }
              size={Math.min(10, Math.max(4, orderedModels.length || 4))}
            >
              {orderedModels.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
            <div className="hint">Hold Ctrl/Cmd to select multiple.</div>
            <div className="hint">Top 10 models are listed first.</div>
          </div>
          <div className="control-block">
            <label>Max Tokens</label>
            <input
              type="number"
              min="1"
              value={runConfig.maxTokens}
              onChange={(e) => setRunConfig((prev) => ({ ...prev, maxTokens: e.target.value }))}
            />
          </div>
          <div className="control-block">
            <label>Temperatures</label>
            <div className="temperature-editor">
              <div className="temperature-chips">
                {runConfig.temperatures.map((value) => (
                  <button
                    key={value}
                    type="button"
                    className="chip secondary"
                    onClick={() => removeTemperatureToken(value)}
                    title="Remove"
                  >
                    {value}
                    <span aria-hidden="true"> x</span>
                  </button>
                ))}
              </div>
              <div className="row-buttons">
                <input
                  type="text"
                  value={temperatureInput}
                  onChange={(e) => setTemperatureInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      addTemperatureTokens();
                    }
                  }}
                  placeholder="0.0, 0.7, auto"
                />
                <button type="button" className="secondary" onClick={addTemperatureTokens}>
                  Add
                </button>
              </div>
            </div>
            <div className="hint">Use `auto` for model default temperature.</div>
          </div>
          <div className="control-block">
            <label>API Types</label>
            <select
              multiple
              value={runConfig.apiTypes}
              onChange={(e) => setRunConfig((prev) => ({ ...prev, apiTypes: getSelectValues(e) }))}
              size={Math.min(4, Math.max(2, modelOptions.apiTypes.length || 2))}
            >
              {modelOptions.apiTypes.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </div>
          <div className="control-block">
            <label>Reasoning Efforts</label>
            <select
              multiple
              value={runConfig.reasoningEfforts}
              onChange={(e) =>
                setRunConfig((prev) => ({ ...prev, reasoningEfforts: getSelectValues(e) }))
              }
              size={Math.min(8, Math.max(3, modelOptions.reasoningEfforts.length + 1 || 3))}
            >
              <option value="auto">auto</option>
              {modelOptions.reasoningEfforts.filter((name) => name !== "auto").map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
            <div className="hint">Hold Ctrl/Cmd to select multiple.</div>
            <div className="hint">`auto` uses the model default reasoning effort.</div>
          </div>
          <div className="control-block">
            <label>Param Handling</label>
            <label className="inline-toggle">
              <input
                type="checkbox"
                checked={runConfig.applyParamCorrections}
                onChange={(e) =>
                  setRunConfig((prev) => ({
                    ...prev,
                    applyParamCorrections: e.target.checked,
                  }))
                }
              />
              <span>Auto-correct params</span>
            </label>
            <div className="hint">Enabled: apply compatibility corrections. Disabled: send values as configured.</div>
          </div>
          <div className="control-block config-requests">
            <label>User prompts</label>
            <div className="request-list">
              {runConfig.requests.map((item, idx) => (
                <div key={`req-${idx}`} className="request-row">
                  <input
                    type="text"
                    value={item.label}
                    onChange={(e) => {
                      const value = e.target.value;
                      setRunConfig((prev) => {
                        const next = [...prev.requests];
                        next[idx] = { ...next[idx], label: value };
                        return { ...prev, requests: next };
                      });
                    }}
                    placeholder={`request_${idx + 1}`}
                  />
                  <textarea
                    rows={3}
                    value={item.prompt}
                    onChange={(e) => {
                      const value = e.target.value;
                      setRunConfig((prev) => {
                        const next = [...prev.requests];
                        next[idx] = { ...next[idx], prompt: value };
                        return { ...prev, requests: next };
                      });
                    }}
                    placeholder="Enter a prompt..."
                  />
                  <button
                    type="button"
                    className="secondary"
                    onClick={() =>
                      setRunConfig((prev) => ({
                        ...prev,
                        requests: prev.requests.filter((_, i) => i !== idx),
                      }))
                    }
                    disabled={runConfig.requests.length === 1}
                  >
                    Remove
                  </button>
                </div>
              ))}
              <button
                type="button"
                onClick={() =>
                  setRunConfig((prev) => ({
                    ...prev,
                    requests: [
                      ...prev.requests,
                      {
                        label: `request_${prev.requests.length + 1}`,
                        prompt: "",
                      },
                    ],
                  }))
                }
              >
                Add Prompt
              </button>
            </div>
          </div>
          </div>
        ) : null}
        {!serverAvailable ? (
          <div className="status error">Server unavailable. Start the API to submit runs.</div>
        ) : null}
      </section>

      <section className="outputs-panel">
        <div className="panel-header">
          <div>
            <h2>Results</h2>
            <p>Load saved runs or upload a JSON file to inspect outputs.</p>
          </div>
        </div>
        <div className="controls">
        <div className="control-block">
          <label>Folder</label>
          <input type="text" value={OUTPUT_DIR} readOnly />
        </div>
        <div className="control-block">
          <label>Saved Runs</label>
          <select
            value={selectedFile}
            disabled={!serverAvailable}
            onChange={(e) => {
              setSelectedFile(e.target.value);
            }}
          >
            <option value="">Select a file...</option>
            {files.map((f) => (
              <option key={f.name} value={f.name}>
                {f.name}
              </option>
            ))}
          </select>
          <div className="hint">Auto-refreshes every 5 seconds. Selecting a file loads it immediately.</div>
          {!serverAvailable ? (
            <div className="status">Server is offline. Use Upload JSON instead.</div>
          ) : null}
        </div>
        <div className="control-block">
          <label>Upload JSON</label>
          <input type="file" accept="application/json" onChange={handleFileUpload} />
        </div>
        <div className="control-block">
          <label>Status</label>
          <div className={error ? "status error" : "status"}>{status}</div>
          {error ? <div className="status error">{error}</div> : null}
        </div>
        </div>
      </section>

      <div className="toolbar">
        <button
          className={forcedOpen === "all" ? "" : "secondary"}
          onClick={() => setAllOpen("all")}
        >
          Expand All
        </button>
        <button className="secondary" onClick={() => setAllOpen("none")}>Collapse All</button>
        <button className="secondary" onClick={() => setOpenMap({})}>Reset Toggles</button>
        <button
          className={compactMode ? "" : "secondary compact-off"}
          onClick={() => setCompactMode((prev) => !prev)}
        >
          Compact Mode
        </button>
      </div>

      {normalized.length === 0 ? (
        <div className="empty">Load a run file to view prompts and outputs.</div>
      ) : (
        <div className="grid">
          {displayOrder.map((idx) => {
            const { entry, index } = normalized.find((item) => item.index === idx) || {};
            if (!entry) return null;
            const call = entry?.llm_call_config || {};
            const sid = entry?.llm_request?.sid ?? call?.sid ?? index + 1;
            const model = call?.model_name || entry?.llm_request?.payload?.model;
            const endpoint = entry?.llm_request?.endpoint;
            const temperature = call?.temperature;
            return (
              <article
                className={`card ${dragIndex === index ? "dragging" : ""} ${dragOverIndex === index && dragIndex !== index ? "drop-target" : ""}`}
                key={`card-${index}`}
                onDragOver={(e) => {
                  e.preventDefault();
                  if (dragIndex !== null && dragIndex !== index) {
                    setDragOverIndex(index);
                  }
                }}
                onDrop={() => {
                  if (dragIndex === null) return;
                  const targetIndex = dragOverIndex ?? index;
                  if (targetIndex === null || dragIndex === targetIndex) {
                    setDragIndex(null);
                    setDragOverIndex(null);
                    return;
                  }
                  setOrder((prev) => reorderList(prev, dragIndex, targetIndex));
                  setDragIndex(null);
                  setDragOverIndex(null);
                }}
              >
                <div className="card-header">
                  <div>
                    <div className="sid">Experiment ID {sid}</div>
                    <div className="meta-row">
                      {model ? <span className="pill">{model}</span> : null}
                      {endpoint ? <span className="pill">{endpoint}</span> : null}
                      {temperature !== undefined ? (
                        <span className="pill">temp {temperature === null ? "null" : temperature}</span>
                      ) : null}
                    </div>
                  </div>
                  <button
                    type="button"
                    className="card-drag"
                    draggable
                    onDragStart={(e) => {
                      e.dataTransfer.effectAllowed = "move";
                      e.dataTransfer.setData("text/plain", "");
                      setDragIndex(index);
                      setDragOverIndex(index);
                    }}
                    onDragEnd={() => {
                      setDragIndex(null);
                      setDragOverIndex(null);
                    }}
                  >
                    Drag to reorder
                  </button>
                </div>
                {compactMode ? (
                  <div className="response-grid">
                    <div className="response-field">
                      <div className="response-label">user prompt</div>
                      {renderValue(getUserPromptText(entry) || "(no user prompt found)")}
                    </div>
                    <div className="response-field">
                      <div className="response-label">output</div>
                      {renderMarkdownOutput(entry?.llm_response?.output)}
                    </div>
                  </div>
                ) : (
                  <>
                    {renderGroup(index, "llm_call_config", entry?.llm_call_config)}
                    {renderGroup(index, "llm_request", entry?.llm_request)}
                    {renderGroup(index, "llm_response", entry?.llm_response)}
                    {Object.entries(entry || {})
                      .filter(([key]) => !["llm_call_config", "llm_request", "llm_response"].includes(key))
                      .map(([key, value]) => renderGroup(index, key, value))}
                  </>
                )}
              </article>
            );
          })}
        </div>
      )}
    </div>
  );
}
