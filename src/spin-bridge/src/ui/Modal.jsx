import { X } from "lucide-react";

export const Modal = ({ isOpen, onClose, title, children }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-brand-dark/60 backdrop-blur-sm">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-2xl max-h-[80vh] flex flex-col overflow-hidden">
        <header className="p-6 border-b border-slate-100 flex justify-between items-center">
          <h3 className="font-montserrat font-bold text-xl text-brand-dark">
            {title}
          </h3>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-brand-orange transition-colors"
          >
            <X size={24} />
          </button>
        </header>
        <div className="p-6 overflow-y-auto font-work-sans">{children}</div>
      </div>
    </div>
  );
};
