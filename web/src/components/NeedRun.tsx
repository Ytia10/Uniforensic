import { useApp } from "../state";
import { Button, Empty } from "./ui";

export default function NeedRun() {
  const { go } = useApp();
  return (
    <Empty>
      <p className="mb-1 font-semibold text-ink">No phone is on the bench yet.</p>
      <p className="mb-5">Choose a case in step 1 and analyse it; this page then shows that phone's results.</p>
      <Button onClick={() => go("intake")}>Open case intake</Button>
    </Empty>
  );
}
