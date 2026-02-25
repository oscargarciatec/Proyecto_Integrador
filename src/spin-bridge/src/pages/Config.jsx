import { useState } from "react";

const Config = () => {
  const [prompt, setPrompt] = useState("Eres un asistente experto en...");
  const [isSaving, setIsSaving] = useState(false);

  const handleSave = async () => {
    setIsSaving(True);
    setTimeout(() => setIsSaving(false), 1000); // Simulación
  };

  return (
    <div className="p-8 space-y-8 bg-[#F8FAFC] min-h-screen">
      <header className="border-b border-brand-primary/20 pb-4">
        <h2 className="text-3xl font-montserrat font-bold text-brand-dark">
          Configuración del Agente
        </h2>
        <p className="text-slate-500 font-work-sans text-sm">
          Modifica el comportamiento del sistema (Prompt Engineering)
        </p>
      </header>

      <div className="bg-white p-8 rounded-2xl shadow-sm border border-slate-100 max-w-4xl">
        <label className="block font-montserrat font-semibold text-brand-dark mb-4">
          Agent Priming (System Prompt)
        </label>

        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          className="w-full h-64 p-4 border border-slate-200 rounded-xl focus:ring-2 focus:ring-brand-primary focus:border-transparent font-work-sans text-slate-700 outline-none transition-all"
          placeholder="Escribe las instrucciones del bot aquí..."
        />

        <div className="mt-6 flex justify-end">
          <button
            onClick={handleSave}
            disabled={isSaving}
            className={`px-8 py-3 rounded-full font-montserrat font-bold text-white transition-all shadow-lg 
              ${isSaving ? "bg-slate-400" : "bg-brand-orange hover:bg-brand-orange/90 active:scale-95"}`}
          >
            {isSaving ? "Guardando Versión..." : "Actualizar Agente"}
          </button>
        </div>
      </div>
    </div>
  );
};

export default Config;
