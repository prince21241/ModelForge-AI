type ComingSoonProps = {
  title: string;
  body: string;
};

export function ComingSoon({ title, body }: ComingSoonProps) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-8 shadow-sm">
      <p className="text-xs font-semibold uppercase tracking-wide text-blue-700">Coming soon</p>
      <h2 className="mt-2 text-2xl font-semibold text-slate-900">{title}</h2>
      <p className="mt-3 max-w-2xl text-slate-600">{body}</p>
    </section>
  );
}
