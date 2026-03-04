import { useState, useEffect } from "react";
import { useApi } from "../hooks/useApi";
import axios from "axios";
import { LoadingState } from "../ui/LoadingState";
import {
  Save,
  Terminal,
  ShieldAlert,
  Code,
  Activity,
  Edit3,
  X,
  History,
  Eye,
  Calendar,
  Copy,
  Check,
} from "lucide-react";

const Agents = () => {
  const { data: agent, loading } = useApi("/api/agents/current");
  const { data: historyData } = useApi("/api/agents/history"); // Hook para traer las 3 versiones

  const [prompt, setPrompt] = useState("");
  const [description, setDescription] = useState("");
  const [saving, setSaving] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);

  // Estados para el Historial
  const [isHistoryModalOpen, setIsHistoryModalOpen] = useState(false);
  const [selectedVersion, setSelectedVersion] = useState(null);
  const [copiedField, setCopiedField] = useState(null);

  useEffect(() => {
    if (agent) {
      setPrompt(agent.priming || "");
      setDescription(agent.description || "");
    }
  }, [agent]);

  const handlePublish = async () => {
    setSaving(true);
    try {
      const payload = {
        kh_agent: agent?.kh_agent,
        name: agent?.name,
        description: description,
        url: agent?.url,
        agent_definition: agent?.agent_definition || {},
        agent_examples: agent?.agent_examples || {},
        is_supervisor: agent?.is_supervisor ?? false,
        priming: prompt,
      };

      const response = await axios.post(
        "http://127.0.0.1:8000/api/agents/update",
        payload,
      );

      alert(`Nueva versión publicada.\n${response.data.status}`);
      setIsModalOpen(false);
      window.location.reload();
    } catch (err) {
      if (err.response?.status === 400) {
        alert(`⚠️ Info: ${err.response.data.detail}`);
      } else {
        alert("Error crítico al intentar publicar. Revisa la consola.");
      }
    } finally {
      setSaving(false);
    }
  };

  const copyToClipboard = (text, field) => {
    navigator.clipboard.writeText(text);
    setCopiedField(field);
    setTimeout(() => setCopiedField(null), 2000);
  };

  if (loading) return <LoadingState />;
  if (!agent)
    return (
      <div className="p-10 text-center">
        No se encontró la configuración del agente.
      </div>
    );

  return (
    <div className="p-8 max-w-6xl mx-auto space-y-6">
      {/* Header con Descripción Editable */}
      <header className="bg-white dark:bg-brand-primary/10 p-8 rounded-3xl shadow-sm border border-slate-100 dark:border-slate-700 flex justify-between items-center">
        <div className="flex gap-6 items-center">
          <div className="w-16 h-16 bg-brand-primary/10 dark:bg-brand-primary/5 rounded-2xl flex items-center justify-center text-brand-primary dark:text-slate-300">
            <Activity size={32} />
          </div>
          <div>
            <div className="flex items-center gap-3">
              <h2 className="text-3xl font-montserrat font-bold text-brand-dark dark:text-slate-300">
                {agent.name}
              </h2>
            </div>
            <p className="text-slate-500 font-work-sans mt-1 dark:text-slate-400">
              {agent.description || "Sin descripción"}
            </p>
          </div>
        </div>

        <button
          onClick={() => setIsModalOpen(true)}
          className="flex items-center gap-2 bg-brand-orange text-white px-6 py-3 rounded-xl font-bold hover:bg-brand-dark/70 dark:hover:bg-brand-orange/80 transition-all shadow-lg"
        >
          <Edit3 size={18} /> Editar Agente
        </button>
      </header>

      {/* VISTA PREVIA DEL PROMPT ACTUAL*/}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <section className="bg-white dark:bg-brand-primary/10 p-6 rounded-3xl border border-slate-100 dark:border-slate-700 shadow-sm">
            <h3 className="text-sm font-bold text-brand-dark dark:text-slate-300 mb-4 uppercase flex items-center gap-2">
              <Terminal size={16} /> System Prompt Actual
            </h3>
            <div className="bg-brand-dark/5 dark:bg-brand-dark/50 p-6 rounded-2xl border border-slate-100 dark:border-slate-700 max-h-96 overflow-y-auto">
              <p className="font-mono text-sm text-slate-600 dark:text-slate-400 whitespace-pre-wrap leading-relaxed italic">
                {agent.priming}
              </p>
            </div>
          </section>
        </div>

        {/* Panel Lateral */}
        <aside className="space-y-6">
          <div className="bg-brand-primary/5 dark:bg-brand-primary/30 p-6 rounded-2xl border border-brand-primary/10">
            <h4 className="font-montserrat font-bold text-brand-dark dark:text-slate-300 text-sm mb-3 flex items-center gap-2">
              <ShieldAlert
                size={26}
                className="text-brand-primary dark:text-slate-300"
              />{" "}
              Tips para definir un buen prompt
            </h4>
            <ul className="text-sm text-slate-600 dark:text-slate-200 space-y-3 font-work-sans list-disc">
              <li>
                Define el <strong>rol</strong> (ej: Experto en SAT).
              </li>
              <li>
                Brinda <strong>contexto y antecedentes</strong>.
              </li>
              <li>
                Define <strong>tareas claras y específicas</strong>.
              </li>
              <li>
                Establece restricciones y <strong>reglas de seguridad</strong>.
              </li>
              <li>
                Especifica el <strong>idioma</strong> de respuesta.
              </li>
              <li>Proporciona ejemplos para consistencia.</li>
            </ul>
          </div>
        </aside>
      </div>

      {/* SECCIÓN DE CONFIGURACIÓN TÉCNICA E HISTORIAL */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-white dark:bg-brand-primary/10 p-6 rounded-2xl border border-slate-100 dark:border-slate-700 shadow-sm">
          <h3 className="text-sm font-bold text-brand-dark dark:text-slate-300 mb-4 font-montserrat uppercase flex items-center gap-2">
            <Code
              size={16}
              className="text-brand-primary dark:text-slate-300"
            />{" "}
            Configuración Técnica (Read-Only)
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {agent.agent_definition &&
              Object.entries(agent.agent_definition).map(([key, value]) => (
                <div
                  key={key}
                  className="p-3 bg-slate-50 dark:bg-brand-primary/10 rounded-xl border border-slate-100 dark:border-slate-700"
                >
                  <p className="text-[10px] text-slate-400 dark:text-slate-300 font-bold uppercase mb-1">
                    {key}
                  </p>
                  <p className="text-xs text-brand-dark dark:text-slate-200 font-mono break-all leading-relaxed">
                    {typeof value === "object"
                      ? JSON.stringify(value)
                      : String(value)}
                  </p>
                </div>
              ))}
          </div>
        </div>

        {/* SECCIÓN DE HISTORIAL RÁPIDO */}
        <div className="bg-white dark:bg-brand-primary/10 p-6 rounded-2xl border border-slate-100 dark:border-slate-700 shadow-sm">
          <h3 className="text-sm font-bold text-brand-dark dark:text-slate-300 mb-4 font-montserrat uppercase flex items-center gap-2">
            <History
              size={16}
              className="text-brand-primary dark:text-slate-300"
            />{" "}
            Historial Reciente
          </h3>
          <div className="space-y-3">
            {historyData?.length > 0 ? (
              historyData.map((rev, idx) => (
                <div
                  key={idx}
                  onClick={() => {
                    setSelectedVersion(rev);
                    setIsHistoryModalOpen(true);
                  }}
                  className="group flex items-center justify-between p-3 bg-slate-50 dark:bg-brand-primary/10 rounded-xl border border-transparent hover:border-brand-primary/30 dark:hover:border-brand-primary cursor-pointer transition-all"
                >
                  <div className="flex items-center gap-3">
                    <Calendar
                      size={18}
                      className="text-slate-400 dark:text-slate-300"
                    />
                    <span className="text-sm font-bold text-slate-600 dark:text-slate-200">
                      {rev.date}
                    </span>
                  </div>
                  <Eye
                    size={14}
                    className="text-slate-300 group-hover:text-brand-primary dark:text-slate-200 dark:group-hover:text-brand-primary"
                  />
                </div>
              ))
            ) : (
              <p className="text-xs text-slate-400 italic dark:text-slate-200">
                No hay versiones previas.
              </p>
            )}
          </div>
        </div>
      </div>

      {/* MODAL DE EDICIÓN (ORIGINAL) */}
      {isModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-brand-dark/60 backdrop-blur-sm">
          <div className="bg-white dark:bg-brand-dark w-full max-w-4xl rounded-3xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
            <div className="p-6 border-b flex justify-between items-center bg-slate-50 dark:bg-brand-primary/20">
              <div className="flex items-center gap-3 text-brand-dark dark:text-slate-300">
                <div className="p-2 bg-brand-orange/10 rounded-lg text-brand-orange dark:bg-brand-orange/5">
                  <Edit3 size={20} />
                </div>
                <div>
                  <h3 className="font-montserrat font-bold">
                    Configurar {agent.name}
                  </h3>
                  <p className="text-[10px] text-slate-400 uppercase font-bold tracking-tighter">
                    Nueva Versión del Agente
                  </p>
                </div>
              </div>
              <button
                onClick={() => setIsModalOpen(false)}
                className="p-2 hover:bg-slate-200 rounded-full transition-colors"
              >
                <X size={20} />
              </button>
            </div>
            <div className="p-8 overflow-y-auto space-y-6">
              <div className="space-y-2">
                <label className="text-xs font-bold text-slate-400 uppercase ml-1">
                  Descripción del Agente
                </label>
                <input
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  className="w-full p-4 bg-slate-50  dark:bg-brand-primary/10 dark:text-slate-300 rounded-xl border border-slate-200 dark:border-slate-700 focus:border-brand-primary outline-none font-work-sans transition-all "
                />
              </div>
              <div className="space-y-2">
                <label className="text-xs font-bold text-slate-400 dark:text-slate-300 uppercase">
                  System Prompt (Priming)
                </label>
                <textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  className="w-full h-[400px] p-6 bg-brand-dark/5 dark:bg-brand-primary/10 text-slate-600 dark:text-slate-200 font-mono text-sm rounded-2xl border border-slate-800 dark:border-slate-700 outline-none resize-none focus:ring-2 ring-brand-primary/20 dark:ring-brand-primary/20 transition-all shadow-inner"
                />
              </div>
            </div>
            <div className="p-6 border-t bg-slate-50 dark:bg-brand-primary/5 flex justify-end gap-3">
              <button
                onClick={() => setIsModalOpen(false)}
                className="px-6 py-3 rounded-xl font-bold text-slate-500 dark:text-slate-300"
              >
                Cancelar
              </button>
              <button
                onClick={handlePublish}
                disabled={saving}
                className="px-8 py-3 bg-brand-orange text-white rounded-xl font-bold flex items-center gap-2 hover:bg-brand-orange/90 transition-all shadow-lg shadow-brand-orange/20 disabled:opacity-50"
              >
                <Save size={20} /> {saving ? "Guardando..." : "Guardar Cambios"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* MODAL DE HISTORIAL (NUEVO) */}
      {isHistoryModalOpen && selectedVersion && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-brand-dark/60  backdrop-blur-sm">
          <div className="bg-white dark:bg-brand-dark w-full max-w-4xl rounded-3xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
            <div className="p-6 border-b flex justify-between items-center bg-brand-primary/5 dark:bg-brand-primary/20">
              <div className="flex items-center gap-3">
                <History className="text-brand-primary dark:text-slate-300" />
                <div>
                  <h3 className="font-montserrat font-bold text-brand-dark dark:text-slate-300">
                    Versión: {selectedVersion.date}
                  </h3>
                  <p className="text-[10px] text-slate-400 uppercase font-mono dark:text-slate-200"></p>
                </div>
              </div>
              <button
                onClick={() => setIsHistoryModalOpen(false)}
                className="p-2 hover:bg-slate-200 rounded-full"
              >
                <X size={20} />
              </button>
            </div>

            <div className="p-8 overflow-y-auto space-y-6">
              {/* Descripción Histórica */}
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <label className="text-xs font-bold text-slate-400 dark:text-slate-300 uppercase ml-1">
                    Descripción de esta versión
                  </label>
                  <button
                    onClick={() =>
                      copyToClipboard(selectedVersion.description, "desc")
                    }
                    className="flex items-center gap-1 text-[10px] font-bold text-brand-primary dark:text-slate-300 hover:underline"
                  >
                    {copiedField === "desc" ? (
                      <Check size={16} />
                    ) : (
                      <Copy size={16} />
                    )}
                    {copiedField === "desc" ? "Copiado" : "Copiar Descripción"}
                  </button>
                </div>
                <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 dark:bg-brand-primary/10 dark:text-slate-300 text-sm italic text-slate-600">
                  {selectedVersion.description}
                </div>
              </div>

              {/* Prompt Histórico */}
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <label className="text-xs font-bold text-slate-400 dark:text-slate-300 uppercase ml-1">
                    System Prompt Histórico
                  </label>
                  <button
                    onClick={() =>
                      copyToClipboard(selectedVersion.priming, "prompt")
                    }
                    className="flex items-center gap-1 text-[10px] font-bold text-brand-primary dark:text-slate-300 hover:underline"
                  >
                    {copiedField === "prompt" ? (
                      <Check size={16} />
                    ) : (
                      <Copy size={16} />
                    )}
                    {copiedField === "prompt" ? "Copiado" : "Copiar Prompt"}
                  </button>
                </div>
                <div className="bg-brand-dark/5 dark:bg-brand-primary/10 p-6 rounded-2xl border border-slate-800 h-80 overflow-y-auto">
                  <pre className="text-xs text-slate-600 dark:text-slate-200 font-mono whitespace-pre-wrap">
                    {selectedVersion.priming}
                  </pre>
                </div>
              </div>
            </div>

            <div className="p-6 border-t bg-slate-50 dark:bg-brand-primary/5 flex justify-end">
              <button
                onClick={() => setIsHistoryModalOpen(false)}
                className="px-8 py-3 bg-brand-orange hover:bg-brand-dark/70 text-white rounded-xl font-bold transition-all"
              >
                Cerrar Vista
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Agents;
