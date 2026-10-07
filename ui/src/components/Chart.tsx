import { useEffect, useRef } from "react";
import * as echarts from "echarts/core";
import { BarChart } from "echarts/charts";
import { GridComponent, MarkLineComponent, TooltipComponent } from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import type { EChartsCoreOption } from "echarts/core";

echarts.use([BarChart, GridComponent, TooltipComponent, MarkLineComponent, CanvasRenderer]);

interface Props {
  option: EChartsCoreOption;
  className?: string;
  height?: number;
  onBarClick?: (dataIndex: number) => void;
  ariaLabel: string;
}

/** Thin ECharts wrapper: one instance per mount, resized with its container, options replaced wholesale. */
export function Chart({ option, className = "chart", height, onBarClick, ariaLabel }: Props) {
  const el = useRef<HTMLDivElement>(null);
  const inst = useRef<echarts.ECharts | null>(null);
  const clickRef = useRef(onBarClick);
  clickRef.current = onBarClick;

  useEffect(() => {
    const chart = echarts.init(el.current!, undefined, { renderer: "canvas" });
    inst.current = chart;
    chart.on("click", (p: any) => clickRef.current?.(p.dataIndex));
    const ro = new ResizeObserver(() => chart.resize());
    ro.observe(el.current!);
    return () => {
      ro.disconnect();
      chart.dispose();
      inst.current = null;
    };
  }, []);

  useEffect(() => {
    inst.current?.setOption(option, { notMerge: true, lazyUpdate: true });
  }, [option]);

  return <div ref={el} className={className} style={height ? { height } : undefined} role="img" aria-label={ariaLabel} />;
}
