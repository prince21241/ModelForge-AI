import { useEffect, useRef, useState, type ReactNode } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Tooltip, XAxis, YAxis } from "recharts";
import type { ComparisonPoint, SinglePoint } from "../benchmarkMetrics";

const TEACHER_COLOR = "#2563EB";
const STUDENT_COLOR = "#0D9488";

function useChartSize(fallback: { width: number; height: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState(fallback);

  useEffect(() => {
    const node = ref.current;
    if (!node) {
      return;
    }
    const measure = () => {
      const width = node.clientWidth;
      const height = node.clientHeight;
      if (width > 0 && height > 0) {
        setSize({ width, height });
      }
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  return { ref, size };
}

function ChartFrame({
  testId,
  label,
  heightClass,
  fallback,
  children,
}: {
  testId: string;
  label: string;
  heightClass: string;
  fallback: { width: number; height: number };
  children: (size: { width: number; height: number }) => ReactNode;
}) {
  const { ref, size } = useChartSize(fallback);
  return (
    <div className="mt-4 max-w-full overflow-x-auto" data-testid={testId}>
      <div ref={ref} className={`${heightClass} w-full min-w-[20rem]`} role="img" aria-label={label}>
        {children(size)}
      </div>
    </div>
  );
}

type ChartTooltipProps = {
  active?: boolean;
  payload?: Array<{ name?: string; value?: number | string | null; color?: string }>;
  label?: string;
  unit?: string;
};

export function ChartTooltip({ active, payload, label, unit = "" }: ChartTooltipProps) {
  if (!active || !payload || payload.length === 0) {
    return null;
  }
  return (
    <div className="rounded-xl border border-slate-200 bg-white px-3 py-2 shadow-lg">
      <p className="text-xs font-semibold text-slate-900">{label}</p>
      <ul className="mt-1 space-y-1">
        {payload.map((item) => (
          <li key={String(item.name)} className="text-xs" style={{ color: item.color }}>
            {item.name}: {item.value === undefined || item.value === null ? "Not available" : `${item.value}${unit}`}
          </li>
        ))}
      </ul>
    </div>
  );
}

function hasPairValue(data: ComparisonPoint[]): boolean {
  return data.some((point) => point.teacher !== null || point.student !== null);
}

function hasSingleValue(data: SinglePoint[]): boolean {
  return data.some((point) => point.value !== null);
}

function PairTable({ data, testId }: { data: ComparisonPoint[]; testId: string }) {
  return (
    <div className="mt-4 max-w-full overflow-x-auto">
      <table className="w-full min-w-[20rem] text-left text-sm" data-testid={testId}>
        <thead className="text-slate-500">
          <tr>
            <th className="py-2 pr-3 font-medium" scope="col">
              Measure
            </th>
            <th className="py-2 pr-3 font-medium text-blue-700" scope="col">
              Teacher
            </th>
            <th className="py-2 font-medium text-teal-700" scope="col">
              Student
            </th>
          </tr>
        </thead>
        <tbody>
          {data.map((point) => (
            <tr key={point.key} className="border-t border-slate-100">
              <th className="py-2 pr-3 font-medium text-slate-800" scope="row">
                {point.label}
              </th>
              <td className="py-2 pr-3 text-slate-700">{point.teacherLabel}</td>
              <td className="py-2 text-slate-700">{point.studentLabel}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ComparisonChart({
  title,
  description,
  data,
  unit,
  testId,
  domain,
}: {
  title: string;
  description: string;
  data: ComparisonPoint[];
  unit: string;
  testId: string;
  domain?: [number, number | "auto"];
}) {
  const plotted = data.map((point) => ({
    label: point.label,
    Teacher: point.teacher,
    Student: point.student,
  }));
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <h3 className="text-base font-semibold text-slate-900">{title}</h3>
      <p className="mt-1 text-sm leading-6 text-slate-600">{description}</p>
      {data.length === 0 || !hasPairValue(data) ? (
        <p className="mt-4 text-sm text-slate-600">Not available</p>
      ) : (
        <ChartFrame testId={testId} label={title} heightClass="h-72" fallback={{ width: 640, height: 288 }}>
          {(size) => (
            <BarChart width={size.width} height={size.height} data={plotted} barGap={6} barCategoryGap="28%">
              <CartesianGrid stroke="#e2e8f0" vertical={false} />
              <XAxis dataKey="label" tick={{ fill: "#475569", fontSize: 12 }} interval={0} />
              <YAxis
                domain={domain ?? [0, "auto"]}
                tick={{ fill: "#64748b", fontSize: 12 }}
                width={48}
                allowDecimals={false}
              />
              <Tooltip cursor={{ fill: "rgba(15, 23, 42, 0.04)" }} content={<ChartTooltip unit={unit} />} />
              <Legend />
              <Bar dataKey="Teacher" fill={TEACHER_COLOR} radius={[4, 4, 0, 0]} maxBarSize={42} />
              <Bar dataKey="Student" fill={STUDENT_COLOR} radius={[4, 4, 0, 0]} maxBarSize={42} />
            </BarChart>
          )}
        </ChartFrame>
      )}
      {data.length > 0 ? <PairTable data={data} testId={`${testId}-data`} /> : null}
    </section>
  );
}

function SingleTable({ data, testId, heading }: { data: SinglePoint[]; testId: string; heading: string }) {
  return (
    <table className="mt-4 w-full text-left text-sm" data-testid={testId}>
      <thead className="text-slate-500">
        <tr>
          <th className="py-2 pr-3 font-medium" scope="col">
            {heading}
          </th>
          <th className="py-2 font-medium" scope="col">
            Value
          </th>
        </tr>
      </thead>
      <tbody>
        {data.map((point) => (
          <tr key={point.key} className="border-t border-slate-100">
            <th className="py-2 pr-3 font-medium text-slate-800" scope="row">
              {point.label}
            </th>
            <td className="py-2 text-slate-700">{point.text}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function SingleChart({
  title,
  data,
  unit,
  testId,
  color,
  domain,
}: {
  title: string;
  data: SinglePoint[];
  unit: string;
  testId: string;
  color: string;
  domain?: [number, number | "auto"];
}) {
  const plotted = data.map((point) => ({ label: point.label, Value: point.value }));
  return (
    <div>
      <h4 className="text-sm font-semibold text-slate-900">{title}</h4>
      {data.length === 0 || !hasSingleValue(data) ? (
        <p className="mt-3 text-sm text-slate-600">Not available</p>
      ) : (
        <ChartFrame testId={testId} label={title} heightClass="mt-3 h-56" fallback={{ width: 480, height: 224 }}>
          {(size) => (
            <BarChart width={size.width} height={size.height} data={plotted} barCategoryGap="30%">
              <CartesianGrid stroke="#e2e8f0" vertical={false} />
              <XAxis dataKey="label" tick={{ fill: "#475569", fontSize: 12 }} interval={0} />
              <YAxis domain={domain ?? [0, "auto"]} tick={{ fill: "#64748b", fontSize: 12 }} width={44} allowDecimals={false} />
              <Tooltip cursor={{ fill: "rgba(15, 23, 42, 0.04)" }} content={<ChartTooltip unit={unit} />} />
              <Bar dataKey="Value" name={title} fill={color} radius={[4, 4, 0, 0]} maxBarSize={42} />
            </BarChart>
          )}
        </ChartFrame>
      )}
      {data.length > 0 ? <SingleTable data={data} testId={`${testId}-data`} heading="Measure" /> : null}
    </div>
  );
}
