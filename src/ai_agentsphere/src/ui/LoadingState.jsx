export const LoadingState = ({
  message = "Sincronizando con Spin Compass...",
}) => (
  <div className="flex flex-col items-center justify-center p-12 space-y-6 min-h-[400px]">
    <div className="relative">
      {/* Outer ring */}
      <div className="w-16 h-16 border-4 border-brand-primary/10 rounded-full"></div>
      {/* Spinning element */}
      <div className="absolute top-0 left-0 w-16 h-16 border-4 border-transparent border-t-brand-primary rounded-full animate-spin"></div>
      {/* Moving inner circle for extra flair */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-2 h-2 bg-brand-orange rounded-full animate-ping"></div>
    </div>

    <div className="text-center space-y-2">
      <p className="font-montserrat font-bold text-brand-dark dark:text-slate-300 tracking-wide">
        {message}
      </p>
      <div className="flex justify-center gap-1">
        <div className="w-1.5 h-1.5 bg-brand-primary rounded-full animate-bounce [animation-delay:-0.3s]"></div>
        <div className="w-1.5 h-1.5 bg-brand-primary rounded-full animate-bounce [animation-delay:-0.15s]"></div>
        <div className="w-1.5 h-1.5 bg-brand-primary rounded-full animate-bounce"></div>
      </div>
    </div>
  </div>
);
