import React from "react";

export const KpiCard = React.memo(
  ({
    title,
    value,
    variant = "primary",
    fontSize = "large",
    children,
    isLoading = false,
  }) => {
    const variantColors = {
      primary: "border-t-brand-primary dark:border-t-brand-primary",
      purple: "border-t-brand-purple dark:border-t-brand-purple",
      orange: "border-t-brand-orange dark:border-t-brand-orange",
    };

    const sizeClasses = {
      small: "text-xs md:text-sm",
      medium: "text-sm md:text-md",
      large: "text-md md:text-lg",
      xlarge: "text-lg xl:text-xl",
      xxlarge: "text-xl xl:text-2xl",
      xxxlarge: "text-2xl xl:text-4xl",
    };

    return (
      <div
        className={`
          relative p-6 rounded-2xl shadow-sm transition-all duration-300 hover:shadow-md
          bg-white dark:bg-slate-900 
          text-brand-dark dark:text-white
          border border-slate-100 dark:border-slate-800
          border-t-6 ${variantColors[variant]}
          ${isLoading ? "animate-pulse" : ""}
        `}
      >
        <div className="font-work-sans text-[10px] xl:text-[11px] uppercase tracking-widest text-slate-500 dark:text-slate-400 mb-1 font-bold h-4">
          {isLoading ? (
            <div className="w-20 h-full bg-slate-200 dark:bg-slate-800 rounded"></div>
          ) : (
            title
          )}
        </div>
        <div className="flex items-end justify-between min-h-[40px]">
          <h3
            className={`font-poppins font-bold tracking-tight transition-colors ${sizeClasses[fontSize]} flex-1`}
          >
            {isLoading ? (
              <div className="w-3/4 h-8 bg-slate-200 dark:bg-slate-800 rounded mt-2"></div>
            ) : (
              value
            )}
          </h3>
          <div className="shrink-0 transition-transform duration-300 hover:scale-110">
            {isLoading ? (
              <div className="w-8 h-8 bg-slate-200 dark:bg-slate-800 rounded-lg"></div>
            ) : (
              children
            )}
          </div>
        </div>
      </div>
    );
  },
  (prev, next) => {
    return (
      prev.value === next.value &&
      prev.title === next.title &&
      prev.variant === next.variant &&
      prev.fontSize === next.fontSize &&
      prev.isLoading === next.isLoading
    );
  },
);
