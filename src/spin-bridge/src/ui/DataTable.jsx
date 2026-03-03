export const DataTable = ({ columns, data, onRowClick }) => {
  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-100 overflow-hidden">
      <table className="w-full text-left border-collapse">
        <thead className="bg-slate-50/50 border-b border-slate-100">
          <tr>
            {columns.map((col) => (
              <th
                key={col.key}
                className="p-4 font-montserrat font-bold text-brand-dark text-sm uppercase tracking-wider"
              >
                {col.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="font-work-sans divide-y divide-slate-50">
          {data.map((row, index) => (
            <tr
              key={index}
              onClick={() => onRowClick?.(row)}
              className="hover:bg-brand-primary/5 transition-colors cursor-pointer group"
            >
              {columns.map((col) => (
                <td key={col.key} className="p-4 text-slate-600">
                  {col.render ? col.render(row[col.key], row) : row[col.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
