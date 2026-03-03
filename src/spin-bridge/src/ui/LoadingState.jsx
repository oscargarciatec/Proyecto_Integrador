export const LoadingState = () => (
  <div className="flex flex-col items-center justify-center p-20 space-y-4">
    {/* Spinner con color Primario #61B5CC */}
    <div className="w-12 h-12 border-4 border-brand-primary/20 border-t-brand-primary rounded-full animate-spin"></div>
    <p className="font-montserrat font-bold text-brand-dark animate-pulse">
      Sincronizando con Spin Compass...
    </p>
  </div>
);
