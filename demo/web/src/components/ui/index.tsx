import * as React from "react";
import { AlertDialog } from "@base-ui/react/alert-dialog";
import { Menu } from "@base-ui/react/menu";
import { NumberField as BaseNumberField } from "@base-ui/react/number-field";
import { Progress } from "@base-ui/react/progress";
import { Select as BaseSelect } from "@base-ui/react/select";
import { Slider as BaseSlider } from "@base-ui/react/slider";
import { Switch as BaseSwitch } from "@base-ui/react/switch";
import { Tabs as BaseTabs } from "@base-ui/react/tabs";
import { Toggle } from "@base-ui/react/toggle";
import { ToggleGroup } from "@base-ui/react/toggle-group";
import { Tooltip as BaseTooltip } from "@base-ui/react/tooltip";
import { Check, ChevronsUpDown, Minus, Plus } from "lucide-react";
import { cn } from "../../lib/cn";

const focusRing = "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-400";

// ---------------------------------------------------------------------------------------------- Button
type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: "sm" | "md" | "icon";
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = "secondary", size = "md", className, type = "button", ...props },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      className={cn(
        "inline-flex select-none items-center justify-center gap-1.5 rounded-md font-medium whitespace-nowrap transition-colors duration-100 disabled:cursor-not-allowed disabled:opacity-50",
        focusRing,
        size === "md" && "h-9 px-3.5 text-sm",
        size === "sm" && "h-7 px-2.5 text-xs",
        size === "icon" && "size-8",
        variant === "primary" && "bg-blue-600 text-white hover:enabled:bg-blue-500",
        variant === "secondary" && "border border-stone-700 bg-stone-800/60 text-stone-100 hover:enabled:bg-stone-800",
        variant === "ghost" && "text-stone-300 hover:enabled:bg-stone-800 hover:enabled:text-stone-100",
        variant === "danger" && "border border-red-900/70 bg-red-950/40 text-red-200 hover:enabled:bg-red-950/70",
        className,
      )}
      {...props}
    />
  );
});

// ---------------------------------------------------------------------------------------------- Field label
export function FieldLabel({ children, hint, htmlFor }: { children: React.ReactNode; hint?: React.ReactNode; htmlFor?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <label htmlFor={htmlFor} className="text-xs font-medium text-stone-300">
        {children}
      </label>
      {hint && <span className="text-xs text-stone-500 tabular-nums">{hint}</span>}
    </div>
  );
}

// ---------------------------------------------------------------------------------------------- Slider
export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  format = (v) => String(v),
  disabled,
  className,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  disabled?: boolean;
  className?: string;
}) {
  return (
    <BaseSlider.Root
      value={value}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      onValueChange={(v) => onChange(Array.isArray(v) ? v[0] : v)}
      className={cn("flex flex-col gap-1.5", className)}
    >
      <div className="flex items-baseline justify-between gap-2">
        <BaseSlider.Label className="text-xs font-medium text-stone-300">{label}</BaseSlider.Label>
        <span className="text-xs text-stone-400 tabular-nums">{format(value)}</span>
      </div>
      <BaseSlider.Control className="flex h-4 w-full touch-none items-center select-none data-disabled:opacity-50">
        <BaseSlider.Track className="relative h-1 w-full rounded-full bg-stone-700">
          <BaseSlider.Indicator className="rounded-full bg-blue-500" />
          <BaseSlider.Thumb
            aria-label={label}
            className={cn("size-3.5 rounded-full border border-stone-950 bg-stone-100 shadow-sm", "has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-blue-400")}
          />
        </BaseSlider.Track>
      </BaseSlider.Control>
    </BaseSlider.Root>
  );
}

// ---------------------------------------------------------------------------------------------- Switch
export function Switch({
  label,
  checked,
  onChange,
  disabled,
  description,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
  description?: string;
}) {
  return (
    <label className={cn("flex cursor-pointer items-start gap-2.5", disabled && "cursor-not-allowed opacity-50")}>
      <BaseSwitch.Root
        checked={checked}
        disabled={disabled}
        onCheckedChange={(v) => onChange(v)}
        className={cn(
          "relative mt-0.5 flex h-4.5 w-8 shrink-0 rounded-full bg-stone-700 p-0.5 transition-colors duration-100 data-checked:bg-blue-600",
          focusRing,
        )}
      >
        <BaseSwitch.Thumb className="size-3.5 rounded-full bg-white transition-transform duration-100 data-checked:translate-x-3.5" />
      </BaseSwitch.Root>
      <span className="flex flex-col">
        <span className="text-xs font-medium text-stone-200">{label}</span>
        {description && <span className="text-xs text-pretty text-stone-500">{description}</span>}
      </span>
    </label>
  );
}

// ---------------------------------------------------------------------------------------------- NumberField
export function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  hint,
  disabled,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  hint?: string;
  disabled?: boolean;
}) {
  const id = React.useId();
  return (
    <BaseNumberField.Root
      id={id}
      value={value}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      onValueChange={(v) => v !== null && onChange(v)}
      className="flex flex-col gap-1.5"
    >
      <FieldLabel htmlFor={id} hint={hint}>
        {label}
      </FieldLabel>
      <BaseNumberField.Group className="flex h-8 overflow-hidden rounded-md border border-stone-700 bg-stone-950/60 focus-within:border-blue-500">
        <BaseNumberField.Decrement
          aria-label={`Decrease ${label}`}
          className="flex w-7 items-center justify-center text-stone-400 hover:bg-stone-800 hover:text-stone-100 disabled:opacity-40"
        >
          <Minus className="size-3.5" />
        </BaseNumberField.Decrement>
        <BaseNumberField.Input className="w-full min-w-0 bg-transparent text-center text-sm text-stone-100 tabular-nums outline-none" />
        <BaseNumberField.Increment
          aria-label={`Increase ${label}`}
          className="flex w-7 items-center justify-center text-stone-400 hover:bg-stone-800 hover:text-stone-100 disabled:opacity-40"
        >
          <Plus className="size-3.5" />
        </BaseNumberField.Increment>
      </BaseNumberField.Group>
    </BaseNumberField.Root>
  );
}

// ---------------------------------------------------------------------------------------------- Select
export interface SelectItem<T extends string> {
  value: T;
  label: string;
  description?: string;
}

export function Select<T extends string>({
  label,
  value,
  items,
  onChange,
  disabled,
  className,
}: {
  label?: string;
  value: T;
  items: SelectItem<T>[];
  onChange: (v: T) => void;
  disabled?: boolean;
  className?: string;
}) {
  return (
    <BaseSelect.Root
      items={items}
      value={value}
      disabled={disabled}
      onValueChange={(v) => v !== null && onChange(v as T)}
    >
      <div className={cn("flex flex-col gap-1.5", className)}>
        {label && <BaseSelect.Label className="text-xs font-medium text-stone-300">{label}</BaseSelect.Label>}
        <BaseSelect.Trigger
          className={cn(
            "flex h-8 items-center justify-between gap-2 rounded-md border border-stone-700 bg-stone-950/60 pr-2 pl-2.5 text-sm text-stone-100 select-none hover:not-data-disabled:bg-stone-900 data-disabled:opacity-50",
            focusRing,
          )}
        >
          <BaseSelect.Value className="truncate" />
          <BaseSelect.Icon className="text-stone-500">
            <ChevronsUpDown className="size-3.5" />
          </BaseSelect.Icon>
        </BaseSelect.Trigger>
      </div>
      <BaseSelect.Portal>
        <BaseSelect.Positioner className="z-50 outline-none select-none" sideOffset={4} alignItemWithTrigger={false}>
          <BaseSelect.Popup className="min-w-(--anchor-width) origin-(--transform-origin) rounded-md border border-stone-700 bg-stone-900 py-1 text-stone-100 shadow-lg transition-[scale,opacity] duration-100 ease-out data-ending-style:scale-95 data-ending-style:opacity-0 data-starting-style:scale-95 data-starting-style:opacity-0">
            <BaseSelect.List className="max-h-(--available-height) overflow-y-auto">
              {items.map((it) => (
                <BaseSelect.Item
                  key={it.value}
                  value={it.value}
                  className="grid cursor-default grid-cols-[1rem_1fr] items-start gap-2 py-1.5 pr-3 pl-2 text-sm outline-none select-none data-highlighted:bg-stone-800"
                >
                  <BaseSelect.ItemIndicator className="col-start-1 mt-0.5 text-blue-400">
                    <Check className="size-3.5" />
                  </BaseSelect.ItemIndicator>
                  <span className="col-start-2 flex flex-col">
                    <BaseSelect.ItemText>{it.label}</BaseSelect.ItemText>
                    {it.description && <span className="text-xs text-stone-500">{it.description}</span>}
                  </span>
                </BaseSelect.Item>
              ))}
            </BaseSelect.List>
          </BaseSelect.Popup>
        </BaseSelect.Positioner>
      </BaseSelect.Portal>
    </BaseSelect.Root>
  );
}

// ---------------------------------------------------------------------------------------------- Segmented
export function Segmented<T extends string>({
  value,
  items,
  onChange,
  label,
  size = "md",
  className,
}: {
  value: T;
  items: { value: T; label: React.ReactNode; title?: string }[];
  onChange: (v: T) => void;
  label: string;
  size?: "sm" | "md";
  className?: string;
}) {
  return (
    <ToggleGroup
      aria-label={label}
      value={[value]}
      onValueChange={(v) => v.length && onChange(v[0] as T)}
      className={cn("flex rounded-md border border-stone-700 bg-stone-950/60 p-0.5", className)}
    >
      {items.map((it) => (
        <Toggle
          key={it.value}
          value={it.value}
          title={it.title}
          className={cn(
            "flex flex-1 items-center justify-center gap-1 rounded-[5px] px-2 whitespace-nowrap text-stone-400 select-none hover:text-stone-100 data-pressed:bg-stone-700 data-pressed:text-stone-50",
            size === "md" ? "h-7 text-xs" : "h-6 text-xs",
            focusRing,
          )}
        >
          {it.label}
        </Toggle>
      ))}
    </ToggleGroup>
  );
}

// ---------------------------------------------------------------------------------------------- Tabs
export const Tabs = {
  Root: BaseTabs.Root,
  List: function List({ className, ...p }: React.ComponentProps<typeof BaseTabs.List>) {
    return <BaseTabs.List className={cn("relative flex gap-1 border-b border-stone-800", className as string)} {...p} />;
  },
  Tab: function Tab({ className, ...p }: React.ComponentProps<typeof BaseTabs.Tab>) {
    return (
      <BaseTabs.Tab
        className={cn(
          "-mb-px flex h-8 items-center gap-1.5 border-b-2 border-transparent px-2 text-xs font-medium whitespace-nowrap text-stone-400 select-none hover:text-stone-200 data-active:border-blue-500 data-active:text-stone-50 data-disabled:opacity-40",
          focusRing,
          className as string,
        )}
        {...p}
      />
    );
  },
  Panel: function Panel({ className, ...p }: React.ComponentProps<typeof BaseTabs.Panel>) {
    return <BaseTabs.Panel className={cn("outline-none", className as string)} {...p} />;
  },
};

// ---------------------------------------------------------------------------------------------- Tooltip
export const TooltipProvider = BaseTooltip.Provider;

export function Tooltip({ content, children }: { content: React.ReactNode; children: React.ReactElement }) {
  return (
    <BaseTooltip.Root>
      <BaseTooltip.Trigger render={children} />
      <BaseTooltip.Portal>
        <BaseTooltip.Positioner sideOffset={6} className="z-50">
          <BaseTooltip.Popup className="max-w-64 rounded-md border border-stone-700 bg-stone-900 px-2 py-1 text-xs text-pretty text-stone-200 shadow-md">
            {content}
          </BaseTooltip.Popup>
        </BaseTooltip.Positioner>
      </BaseTooltip.Portal>
    </BaseTooltip.Root>
  );
}

// ---------------------------------------------------------------------------------------------- Menu
export function DropdownMenu({
  trigger,
  items,
}: {
  trigger: React.ReactElement;
  items: { label: string; description?: string; onSelect: () => void; disabled?: boolean }[];
}) {
  return (
    <Menu.Root>
      <Menu.Trigger render={trigger} />
      <Menu.Portal>
        <Menu.Positioner className="z-50 outline-none" sideOffset={6} align="end">
          <Menu.Popup className="min-w-56 origin-(--transform-origin) rounded-md border border-stone-700 bg-stone-900 py-1 text-stone-100 shadow-lg outline-none transition-[scale,opacity] duration-100 ease-out data-ending-style:scale-95 data-ending-style:opacity-0 data-starting-style:scale-95 data-starting-style:opacity-0">
            {items.map((it) => (
              <Menu.Item
                key={it.label}
                disabled={it.disabled}
                onClick={it.onSelect}
                className="flex cursor-default flex-col px-3 py-1.5 text-sm outline-none select-none data-disabled:opacity-40 data-highlighted:bg-stone-800"
              >
                <span>{it.label}</span>
                {it.description && <span className="text-xs text-stone-500">{it.description}</span>}
              </Menu.Item>
            ))}
          </Menu.Popup>
        </Menu.Positioner>
      </Menu.Portal>
    </Menu.Root>
  );
}

// ---------------------------------------------------------------------------------------------- Confirm (AlertDialog)
export function Confirm({
  open,
  onOpenChange,
  title,
  description,
  confirmLabel,
  onConfirm,
}: {
  open: boolean;
  onOpenChange: (v: boolean) => void;
  title: string;
  description: string;
  confirmLabel: string;
  onConfirm: () => void;
}) {
  return (
    <AlertDialog.Root open={open} onOpenChange={onOpenChange}>
      <AlertDialog.Portal>
        <AlertDialog.Backdrop className="fixed inset-0 z-40 bg-black/60 transition-opacity duration-150 data-ending-style:opacity-0 data-starting-style:opacity-0" />
        <AlertDialog.Popup className="fixed top-1/2 left-1/2 z-50 w-96 max-w-[calc(100vw-2rem)] -translate-x-1/2 -translate-y-1/2 rounded-lg border border-stone-700 bg-stone-900 p-5 shadow-xl transition-[scale,opacity] duration-150 ease-out data-ending-style:scale-95 data-ending-style:opacity-0 data-starting-style:scale-95 data-starting-style:opacity-0">
          <AlertDialog.Title className="text-base font-semibold text-balance text-stone-50">{title}</AlertDialog.Title>
          <AlertDialog.Description className="mt-2 text-sm text-pretty text-stone-400">{description}</AlertDialog.Description>
          <div className="mt-5 flex justify-end gap-2">
            <AlertDialog.Close render={<Button variant="ghost">Keep it</Button>} />
            <Button
              variant="primary"
              onClick={() => {
                onOpenChange(false);
                onConfirm();
              }}
            >
              {confirmLabel}
            </Button>
          </div>
        </AlertDialog.Popup>
      </AlertDialog.Portal>
    </AlertDialog.Root>
  );
}

// ---------------------------------------------------------------------------------------------- Progress
export function ProgressBar({ value, label }: { value: number | null; label: string }) {
  return (
    <Progress.Root value={value === null ? null : Math.round(value * 100)} aria-label={label} className="w-full">
      <Progress.Track className="h-1 w-full overflow-hidden rounded-full bg-stone-800">
        <Progress.Indicator className="block h-full bg-blue-500 transition-[width] duration-150 data-indeterminate:w-full data-indeterminate:bg-stone-600" />
      </Progress.Track>
    </Progress.Root>
  );
}

// ---------------------------------------------------------------------------------------------- misc
export function Section({ title, children, aside }: { title: string; children: React.ReactNode; aside?: React.ReactNode }) {
  return (
    <section className="flex flex-col gap-3 border-b border-stone-800 px-4 py-4">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-xs font-semibold text-balance text-stone-400 uppercase">{title}</h2>
        {aside}
      </div>
      {children}
    </section>
  );
}

export function ErrorText({ children }: { children: React.ReactNode }) {
  if (!children) return null;
  return (
    <p role="alert" className="rounded-md border border-red-900/60 bg-red-950/30 px-2.5 py-1.5 text-xs text-pretty text-red-200">
      {children}
    </p>
  );
}

export function Hint({ children, className }: { children: React.ReactNode; className?: string }) {
  return <p className={cn("text-xs text-pretty text-stone-500", className)}>{children}</p>;
}
