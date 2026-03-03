export const formatNumber = (num) => new Intl.NumberFormat("es-MX").format(num);

export const formatPercent = (decimal) => `${decimal}%`;

export const truncateMessage = (text, limit = 50) =>
  text?.length > limit ? `${text.substring(0, limit)}...` : text;

export const groupDataByDate = (data) => {
  return data.reduce((acc, item) => {
    const date = new Date(item.ct_valid_from_dt).toLocaleDateString("es-MX");
    acc[date] = (acc[date] || 0) + 1;
    return acc;
  }, {});
};
