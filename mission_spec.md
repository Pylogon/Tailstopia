# Mission Workflow — Final Optimization & Reliability Specification

คุณรับบทเป็น Senior Python Automation Engineer / QA Automation Architect

ช่วยตรวจสอบและปรับปรุงโปรเจกต์ TailsTopia Mission Workflow โดยยึด `note.txt` เป็น Functional Requirement หลัก และใช้ Source Code, Config, Resources และ Tests ปัจจุบันประกอบการวิเคราะห์

เป้าหมายคือปรับปรุง **Reliability, CPU/RAM Efficiency, Retry/Error Handling, Loading Detection, Configurability และ Debuggability** โดยไม่ Rewrite ทั้งโปรเจกต์

## 0. Project Context และ Optimization Priorities

เครื่องที่ใช้รันจริง:

- Windows บน HP Laptop รุ่นเก่า
- CPU Intel i3-6006U
- RAM 4GB
- มีข้อจำกัดด้าน Performance ค่อนข้างมาก

ผู้ใช้ยอมรอเพิ่ม 1–5 วินาที เพื่อให้เครื่องทำงานเบาลงและเสถียรกว่า

**ลำดับความสำคัญ:**

1. Correctness / Reliability / Safe Stop
2. ลด CPU Usage, RAM Usage และจำนวน Screenshot/Capture
3. รักษาความถูกต้องของ Macro Timing
4. ลด Latency เฉพาะเมื่อมีหลักฐานว่าไม่เพิ่มภาระเครื่องอย่างมีนัยสำคัญ

ห้ามเพิ่ม Polling Frequency เพียงเพื่อประหยัดเวลาเล็กน้อยโดยไม่มี Benchmark รองรับ

ให้รักษา Modular Architecture เดิมและทำ Minimal Refactor เป็นหลัก

---

# 1. Final Mission Workflow

ลำดับการทำงานที่ต้องรักษา:

**STEP 1 — VERIFY_START**

- ใช้บริเวณ Time HUD ด้านบนสุดของหน้าจอเป็น Checkpoint
- ตรวจ READY / LOADING ตามหลักการใน `StartDetector`
- ROI นี้ผู้ใช้เลือกเพราะไม่มีตัวละครหรือ Effect บัง
- ห้ามเปลี่ยนไปใช้ Marker บริเวณที่ตัวละครเคลื่อนที่ผ่านโดยไม่มีเหตุผล
- หากยังไม่ READY ให้ Poll ตาม Config จนสำเร็จหรือ Timeout

**STEP 2 — PLAY_START_MACRO**

- Replay `recordings/start.json`
- ใช้ Recorded Timestamp เดิม
- ไม่ต้องใช้ OpenCV ระหว่าง Macro
- ตรวจว่า Macro Replay ส่ง Action ครบ
- ไม่แก้ Timing ภายใน Macro โดยพลการ

**STEP 3 — OPEN_MISSION**

- กด Talk NPC ด้วยพิกัด Client-relative จาก Config
- พิกัดเดิม `[515,313]` สัมพันธ์กับ Desktop `(558,408)` ภายใต้ตำแหน่งอ้างอิงเดิม
- รอ 5.5 วินาทีตาม Config ก่อนตรวจภาพ
- ตรวจว่าปุ่ม Room List ปรากฏหรือยัง
- ถ้ายังไม่พบ ให้ทำ Retry ตาม `note.txt`
- หาก Fixed-position Retry ไม่สำเร็จ ให้ใช้ Talk Template Fallback จากภาพ Resource
- ตรวจ Room List อีกครั้ง
- Retry/Timeout หมดจึง Safe Stop

สำคัญ: Talk Fallback ใช้เฉพาะ State `OPEN_MISSION` ห้ามย้อนกลับไปกด Talk NPC จาก `OPEN_ROOM_LIST` หรือ `CREATE_ROOM` เพราะอาจปิดหน้าต่าง Mission

**STEP 4 — OPEN_ROOM_LIST**

- Entry Condition: ยืนยันว่าหน้าต่าง Mission และ Room List พร้อมแล้ว
- คลิก Room List ด้วย Position-first
- รอตาม Timing Config (ค่าเดิม 1 วินาที)
- ใช้ OpenCV ตรวจว่า Create ปรากฏ
- ถ้าพบ Create ให้ SUCCESS
- ถ้าไม่พบ ให้ตรวจ Target ซ้ำตามนโยบาย Retry
- หากยังไม่สำเร็จ ให้ตรวจ Room List Template เพื่อหาตำแหน่งคลิกใหม่
- คลิกซ้ำเฉพาะเมื่อยืนยันว่า Source เดิมยังอยู่และ Target ยังไม่ปรากฏ
- หากไม่พบทั้ง Source และ Target ให้รอตรวจใหม่ภายใน Budget โดยไม่คลิกสุ่ม
- Retry/Timeout หมดจึง Safe Stop

**STEP 5 — CREATE_ROOM**

ใช้ Logic ร่วมกับ OPEN_ROOM_LIST:

- Position-first คลิก Create
- รอตาม Config
- ใช้ OpenCV ตรวจ Start
- ถ้า Start ปรากฏแล้วให้ SUCCESS
- ถ้าไม่พบให้ Retry/Template Fallback เฉพาะ Create
- ห้ามย้อนกลับไป Talk NPC
- ถ้า Unknown State หรือหมด Budget ให้ Safe Stop

**STEP 6 — START_STAGE**

- ตรวจว่า Start เป็น Expected Source ของ State นี้
- Position-first คลิก Start ตาม Config
- ยืนยันการเปลี่ยนสถานะก่อนเริ่ม Mission Macro
- อย่าคลิก Start ซ้ำเพียงเพราะยังไม่เห็น Loading
- รักษาระยะเวลารอและ Timeout เดิมเป็นค่าเริ่มต้น

**STEP 7 — WAIT_STAGE / VERIFY_STAGE**

ปัญหาของโค้ดเดิมคือ `WAIT_STAGE_LOADING` บังคับให้พบ LOADING ก่อนเสมอ แม้เกมอาจโหลดเร็วจนโปรแกรมตรวจไม่ทัน

ต้องการ Logic:

1. พบ Loading → รอ Stage Ready
2. ไม่พบ Loading แต่ยืนยัน Stage Ready ได้จริง → SUCCESS
3. ไม่พบ Loading และยังยืนยัน Stage Ready ไม่ได้ → Poll ภายใน Timeout
4. Timeout → Safe Stop

ข้อควรระวังสำคัญ:

- Time HUD สามารถใช้ตรวจ READY ได้ แต่ไม่ได้ยืนยันว่าเป็น Camp หรือ Stage หากภาพ HUD เหมือนกัน
- ห้ามถือว่า HUD READY ที่พบทันทีหลังคลิก Start คือ Stage Ready โดยอัตโนมัติ เพราะอาจยังอยู่ Camp
- ตรวจ Resources และ Transition Evidence ที่มีอยู่ก่อนว่าจะยืนยัน Destination ได้อย่างไร
- ต้องมีหลักฐานเพียงพอในการแยกสถานะ หรือออกแบบ Transition Confirmation ที่ปลอดภัย
- หาก Resources ปัจจุบันยังไม่สามารถยืนยันได้ ให้รายงานข้อจำกัดและคงพฤติกรรมที่ปลอดภัยไว้ก่อน ห้าม Implement การข้าม Loading แบบเสี่ยง False Positive
- ไม่เพิ่ม Poll Frequency เพื่อพยายามจับ Loading ให้ทัน
- คง Time HUD ROI ด้านบนสุดเป็นตัวตรวจหลัก

**STEP 8 — PLAY_MISSION_MACRO**

- Replay `recordings/mission.json`
- ใช้ Recorded Timestamp เดิม
- ไม่ใช้ OpenCV ระหว่าง Replay โดยไม่จำเป็น
- บันทึกเวลาเริ่ม Mission Macro เพื่อใช้คำนวณเวลารอ Result
- ตรวจ Action Completion และรักษา Stop/Pause Behavior เดิม

**STEP 9 — WAIT_MISSION_RESULT**

จากประสบการณ์จริง Mission ใช้เวลาประมาณ 75–90 วินาที

ค่าที่ต้องรักษา:

- เริ่มตรวจ Result หลังจากเวลาเริ่ม Mission Macro ผ่านไป 80 วินาที
- Poll ทุก 1 วินาที
- ใช้ OpenCV Template Matching เฉพาะ ROI ของ Mission Result
- ยืนยันผลต่อเนื่องตาม Config เดิม (2 confirmations)
- หากไม่พบให้ตรวจต่อจน Timeout

ไม่ต้องเปลี่ยนเป็นตรวจตั้งแต่ 75 วินาทีในรอบนี้ เพราะผู้ใช้ให้ความสำคัญกับ CPU มากกว่าการรอเพิ่มไม่กี่วินาที

**STEP 10 — COLLECT_REWARDS**

รักษา Logic ปัจจุบัน:

- หลัง Mission Result ปรากฏ รอ 10 วินาที
- ตรวจ Mission Result ก่อนเริ่มรับ Reward
- เก็บ Baseline ของช่องแรกใน Your Items
- คลิก Reward Slot 1 ตาม Position
- ใช้ Image Difference ตรวจว่าช่อง Your Items เปลี่ยนจากว่างเป็นมีไอเทมหรือไม่
- ตรวจยืนยันตามค่า Confirmations เดิม
- หากยังไม่ได้ ให้ Retry ช่องแรกตาม Budget เดิม
- หลัง Slot 1 Verified แล้ว คลิก Reward Slots 2–7 ตามพิกัดเดิม
- รักษา Delay และเงื่อนไขตรวจ Result ก่อนคลิกช่องอื่นตามโค้ดปัจจุบัน

ห้ามเปลี่ยน Logic Reward หรือเพิ่ม Verification รายช่องในรอบนี้

เหตุผล: การรับ Reward เป็นส่วนที่ผู้ใช้ไม่ต้องการเพิ่มความเสี่ยง และ Image Difference ขนาดเล็กอยู่แล้ว ไม่ได้มีหลักฐานว่าเป็น Performance Bottleneck

**STEP 11 — RETURN_CAMP**

- คลิก Mission Camp แบบ Position-first จาก Config เมื่อ Entry Condition ถูกต้อง
- หลังคลิกไม่จำเป็นต้องจับภาพ Loading ให้ทันเสมอ
- ไม่คลิก Mission Camp ซ้ำโดยไม่มีหลักฐานว่ายังอยู่ State เดิม
- ใช้ Retry และ Timeout อย่างปลอดภัย

**STEP 12 — VERIFY_RETURN / CAMP_READY**

ใช้แนวคิดเดียวกับการเข้า Stage:

- พบ Loading → รอ Camp Ready
- ไม่พบ Loading แต่ยืนยัน Camp Ready ได้จริง → SUCCESS
- ไม่พบทั้งคู่ → รอ Poll ตาม Config
- Timeout → Safe Stop

ต้องระวัง Time HUD ที่อาจปรากฏทั้ง Camp และ Stage

เมื่อยืนยันว่ากลับ Camp สำเร็จ:

- เพิ่ม Completed Loop
- ถ้ายังไม่ครบ `max_loops` → เริ่ม VERIFY_START รอบใหม่
- ถ้าครบ → SUCCESS

รักษา `max_loops` ที่กำหนดไว้ ห้ามเปลี่ยนโดยพลการ

---

# 2. Shared UI Action Handler — Position-first

ปรับการกดปุ่ม Talk, Room List, Create, Start และ Mission Camp ให้ใช้แนวทาง Position-first ตามความเหมาะสม โดยออกแบบ Shared Handler ที่ Configurable

**Happy Path:**

1. ยืนยัน Entry Condition จาก State ก่อนหน้า
2. ตรวจว่าปลายทางปรากฏแล้วหรือไม่ เฉพาะจุดที่จำเป็นเพื่อป้องกัน Action ซ้ำ
3. ใช้ Fixed Client Position จาก Config คลิกก่อน
4. รอ Settle Time
5. ใช้ OpenCV ตรวจ Expected Target ROI
6. Target พบ → SUCCESS

**Failure / Retry Path:**

1. Target ยังไม่พบ → รอและตรวจตาม Retry Policy
2. ก่อนคลิกซ้ำให้ตรวจ Target อีกครั้ง
3. ถ้ายังไม่พบ Target → ใช้ Source Template Matching
4. ถ้า Source อยู่ → คลิกตำแหน่งที่ตรวจพบและ Verify Target
5. ถ้าทั้ง Source และ Target ไม่พบ → จัดเป็น Unknown State, รอ Recheck ตาม Budget
6. Retry หมดหรือ Timeout → Safe Stop พร้อม Error/Screenshot

ข้อกำหนด:

- ไม่ใช้ Template Matching เพื่อหาพิกัด Source ก่อนคลิกใน Happy Path โดยไม่จำเป็น
- ห้ามคลิกพิกัดคงที่ซ้ำเมื่อไม่รู้สถานะหน้าจอ
- ไม่ใช้ Fallback ข้าม State โดยพลการ
- Retry Count, Verify Count และ State Timeout ต้องไม่ปะปนกัน
- ใช้ State Deadline ป้องกัน Loop ไม่สิ้นสุด
- รักษา Coordinate System แบบ Client-relative ของ UI Click
- ห้ามเปลี่ยน Absolute Desktop Coordinates ที่ Macro Recordings อาจใช้อยู่
- ตรวจ DPI/Window Geometry ตามระบบเดิม

สามารถใช้ Shared Handler กับ Room List และ Create ก่อน แล้วค่อยปรับ Start/Camp ตาม Entry/Exit Verification ที่มีจริง

---

# 3. Timing Configuration

ต้องการให้ Gameplay Timing ทั้งหมดปรับผ่าน JSON ได้โดยไม่แก้ Python Source

ให้ Audit Timing ที่มีอยู่แล้วก่อน ไม่สร้าง Config ซ้ำซ้อน

**Timing ที่ต้องครอบคลุม:**

- Talk NPC first wait / retry wait / fallback wait
- UI settle
- Retry delay
- Retry count
- Click hold duration
- Poll interval
- State timeout
- Stage entry minimum wait
- Stage loading / verification
- Mission Result delay / poll / confirmations
- Reward settle / retry / confirmations / between-slot delay
- Return Camp loading / verification

ให้จัด Config ตาม Action หรือ State อย่างเป็นระบบ ใช้ชื่อที่อธิบายความหมายได้ชัดเจน

หากต้องย้าย Key หรือเปลี่ยน Schema:

- ปรับ `mission_config.py` ให้ Validate ได้
- เก็บ Default เดิมไว้
- ไม่ทำให้ Resource Paths เสีย
- ไม่เปลี่ยนพฤติกรรมเดิมโดยไม่แจ้ง
- เพิ่ม Tests สำหรับ Config
- แสดงตัวอย่าง Config และความหมายของแต่ละค่า

**ค่าที่ต้องคงในรอบนี้:**

- Talk NPC first wait = 5.5s
- Room List/Create settle = 1s
- Stage minimum entry wait = 10s
- Mission Result start = 80s
- Mission Result poll = 1s
- Reward settle = 10s
- Return Polling = ค่าเดิม
- Confirmation counts และ Retry Budgets = ค่าเดิม เว้นแต่จำเป็นต้องแก้ Defect

อย่าเปลี่ยน Timing เพื่อ Optimize Latency ในรอบนี้

---

# 4. CPU/RAM Optimization Rules

ให้ Optimize แบบ Conservative โดยเน้นลดงานที่ไม่จำเป็น

**ควรทำ:**

- ลด Source Template Matching ที่ไม่จำเป็นก่อนคลิก
- ใช้ Position-first เฉพาะ UI ที่ยืนยันตำแหน่งได้
- Capture เฉพาะ ROI ที่ต้องใช้
- Reuse Templates ที่โหลดไว้แล้ว
- หลีกเลี่ยงการแปลงภาพหลายครั้งโดยไม่จำเป็น
- ไม่ทำ OCR หาก Template/Image Difference ตอบโจทย์อยู่แล้ว
- หลีกเลี่ยง Screenshot เพิ่มสำหรับงาน Performance Measurement
- หลีกเลี่ยง Polling ถี่เกินความจำเป็น
- รักษาระยะ Sleep ที่ผู้ใช้เลือกเพื่อลดภาระเครื่อง

**ไม่ควรทำในรอบนี้:**

- เพิ่ม OpenCV ตลอดการ Replay Macro
- เพิ่มการตรวจ Mission Result ตั้งแต่ 75s
- เปลี่ยน Talk NPC เป็น Poll ทุก 0.2–0.5s
- เพิ่ม Verification ทุกช่อง Reward
- เพิ่ม Continuous CPU/RAM Monitoring
- เปลี่ยนหลาย Timing พร้อมกันโดยไม่มีข้อมูลวัด

Optimization ต้องแยกผลด้าน CPU/RAM, Latency และ Reliability อย่างชัดเจน

---

# 5. Lightweight Performance Timing Log

ต้องการเก็บ Timing สำหรับวิเคราะห์ Performance และ Optimize Config ในอนาคต

**ต้องมี Overhead ต่ำที่สุด**

ใช้ `time.perf_counter()` เพื่อวัดงานที่เกิดขึ้นอยู่แล้ว ไม่เพิ่ม Capture หรือ Recognition เพื่อ Profiling

เก็บอย่างน้อย:

- State Total Duration
- Action/Click Duration
- Total Sleep/Wait Duration
- Capture Count และ Total Capture Duration
- OpenCV Recognition Count และ Total Recognition Duration
- Retry Count
- State Status

ให้ Accumulate Metrics ใน Memory และบันทึก Summary เมื่อ State จบหรือ FAIL

ตัวอย่าง Format:

    State: OPEN_ROOM_LIST
    click_ms: 52
    capture_ms: 43
    recognition_ms: 4
    sleep_ms: 1000
    capture_count: 1
    retry_count: 0
    total_ms: 1110
    status: SUCCESS

ตัวเลขข้างต้นเป็นเพียงตัวอย่าง ไม่ใช่ Benchmark จริง

**ข้อจำกัดด้าน Performance Logging:**

- ไม่ Capture ภาพเพิ่ม
- ไม่เขียน Screenshot ทุก Poll
- ไม่ Log Timing ลง Disk ทุกครั้งที่ Poll
- ไม่วัด CPU/RAM แบบ Continuous
- ไม่เปิด Heavy Profiler ระหว่างเล่น
- ใช้ไฟล์ Log เดิมหากเหมาะสม หลีกเลี่ยงระบบ Logging ซ้ำซ้อน
- บันทึกข้อมูลสำคัญเมื่อเกิด Failure แม้เป็น State ที่มีการ Poll จำนวนมาก

หากการเขียน Log ทุก Recognition ในระบบเดิมมีต้นทุนสูง ให้ตรวจและเสนอการลด I/O โดยยังรักษา Diagnostic Value

รายงาน Performance Log Overhead ตามที่วัดได้จริง หากยังไม่ได้ Benchmark บน HP ให้ระบุว่า Not Measured

---

# 6. Live Logging / Failure Debugging

ปัญหาที่พบคือ Export โปรเจกต์จากเครื่อง HP มา Mac แล้วพบเพียง Dry-run Logs ยังไม่ได้ยืนยัน Live Log ของรอบที่เกิดปัญหา

ให้ Audit:

- Entry Point ที่รันจริง
- Config Path
- Resolved Output Directory
- การสร้าง LiveSession
- จุดที่สร้าง ExecutionLog
- กรณี Exception ก่อน Log ถูกสร้าง
- ความแตกต่าง Dry-run / Live Mode

ต้องแสดง Absolute Log Directory ชัดเจนใน Terminal เมื่อสร้าง LiveSession สำเร็จ ก่อนเริ่ม Flow

เมื่อ FAIL ให้บันทึก:

- State ที่ FAIL
- Source / Expected Target
- Retry Attempt
- Last Recognition Scores
- Failure Reason
- Timeout / Elapsed Time
- Last ROI Screenshot
- Full Client Screenshot หากจับภาพได้อย่างปลอดภัย

รักษา Failure Screenshot ปัจจุบัน และไม่เพิ่ม Screenshot ทุก Poll

ห้ามสรุปว่าไม่มี Live Log เพียงเพราะหาไม่เจอใน Export ต้องตรวจพาธจริงก่อน

---

# 7. Architecture / Compatibility

รักษาโครงสร้างเดิม:

- `tailstopia_mission.py` — Entry Point
- `mission_flow.py` — Workflow State Machine
- `mission_runtime.py` — Runtime / Input / Session
- `mission_vision.py` — Image Recognition
- `mission_config.py` — Config Validation
- `mission_flow.json` — Config
- `note.txt` — Original Functional Requirements

ห้าม Rewrite โค้ดที่ทำงานได้อยู่แล้วโดยไม่มีเหตุผล

ต้องรักษา:

- F8 Start
- F7 Pause/Resume
- F9 Stop/Exit
- Emergency Stop
- Input Release
- Macro Timestamps
- Reward Behavior
- CLI Options เช่น `--dry-run`, `--loops`, `--through`
- `max_loops` ตาม Config

ห้ามเพิ่ม Action ที่ข้าม UI State โดยไม่ตรวจสอบ

---

# 8. Automated Tests / Regression Tests

ใช้ Mock Input, Fixture และ Recorded Screenshots เท่าที่มี

ต้องทดสอบ:

1. Position-first Happy Path
2. Target ปรากฏตามปกติ
3. Target ปรากฏช้า แต่ไม่ต้อง Click ซ้ำ
4. Source ตรวจไม่พบชั่วคราว
5. Position Click ไม่สำเร็จ แต่ Template Fallback สำเร็จ
6. Room List → Create Retry
7. Create → Start Retry
8. Source/Target ไม่พบจน Timeout
9. Talk Fallback ไม่ย้อนมาจาก State อื่น
10. พบ Loading ตามปกติ
11. พลาดภาพ Loading แต่ยืนยัน Stage Ready ได้จริง
12. พลาดภาพ Loading แต่ยืนยัน Camp Ready ได้จริง
13. HUD READY แต่ยังยืนยัน Destination ไม่ได้ ต้องไม่ผ่านผิด State
14. Logging / Screenshot เมื่อ FAIL
15. Retry Budget และ Timeout
16. Macro Replay และ Reward Regression
17. Pause/Stop ระหว่าง Wait และ Retry
18. Invalid Config และ Coordinate Validation

**สำคัญ:** Dry-run Success ไม่ได้ยืนยันว่า Live Game Success

ต้องรายงานให้ชัดว่า Tests ไหนใช้ Mock/Fixture, Tests ไหนใช้ภาพจริง และส่วนไหนยังไม่ได้ทดสอบบน HP

---

# 9. Optional — ทำภายหลังเท่านั้น

หัวข้อนี้ยังไม่ต้อง Implement ในรอบแรก เว้นแต่พบหลักฐานว่าจำเป็น

**OPTIONAL A — Talk NPC Delayed Polling**

พิจารณา Sleep ประมาณ 5s แล้ว Poll ทุก 1s แทน Sleep 5.5s คงที่

ทำเฉพาะเมื่อ Benchmark แสดงว่าช่วยด้าน Reliability และไม่เพิ่ม CPU อย่างมีนัยสำคัญ

**OPTIONAL B — Mission Result Start 75s**

ปัจจุบันคง 80s

ทดลองเริ่มตรวจ 75s เฉพาะเมื่อมี Performance Evidence ว่า Capture เพิ่มอีกประมาณ 5 ครั้งต่อ Mission ไม่กระทบเครื่องและมีประโยชน์จริง

**OPTIONAL C — Reward Verification Slots 2–7**

ไม่เปลี่ยนในรอบแรก

ศึกษาได้ภายหลังหากพบปัญหาของหายหรือคลิกไม่ครบจาก Live Logs

**OPTIONAL D — Fine-tune Polling Intervals**

ทดลองเปลี่ยน Poll Interval ทีละ State หลังมี Timing Benchmark จริงเท่านั้น

ห้าม Optimize หลายค่าในรอบเดียวจนหาสาเหตุ Regression ไม่ได้

**OPTIONAL E — Additional Stage/Camp Marker**

ศึกษาภาพที่แยก Stage กับ Camp ได้อย่างเสถียร

ต้องไม่ใช้บริเวณที่ตัวละครหรือ Animation บังง่าย

ถ้าจำเป็นต่อการแก้ Loading อย่างปลอดภัย ให้ยกระดับเป็น Prerequisite และแจ้งสิ่งที่ต้องเพิ่มก่อน Implement ไม่ใช้ Marker สมมติ

**OPTIONAL F — Macro Replay Micro-optimization**

ทำเฉพาะเมื่อพบ Evidence ว่า Dispatch/Input Logging หรือ Replay Engine เป็น Bottleneck

ห้ามเร่ง Recorded Timestamps หรือเปลี่ยน Input Semantics เพื่อหวังลดเวลา

---

# 10. Implementation Order

**Phase 1 — Audit & Verification**

- ตรวจโค้ด Config และ Resources ปัจจุบัน
- เปรียบเทียบกับ `note.txt`
- ตรวจความถูกต้องของ Position และ ROI
- ตรวจ Live Logging Paths
- ระบุข้อจำกัดของ Stage/Camp Destination Verification
- สรุป Root Cause ที่ยืนยันได้และข้อสันนิษฐานที่ยังไม่มีหลักฐาน

**Phase 2 — Reliability Fixes**

- แก้ `advance()` Retry / Error Handling
- ทำ Position-first สำหรับ Room List และ Create
- ขยาย Shared Handler ไปยังปุ่มอื่นเฉพาะเมื่อ Entry/Exit Condition รองรับ
- แก้ Loading Detection แบบปลอดภัยเท่าที่ Resources ยืนยันได้
- ห้ามเพิ่ม Unknown State Auto-click

**Phase 3 — Config & Lightweight Metrics**

- Refactor Timing Configuration
- เพิ่ม Config Validation
- เพิ่ม Lightweight Timing Summary
- ปรับ Live Log Path Visibility
- รักษา Failure Screenshots

**Phase 4 — Tests / Regression**

- Unit Tests
- Mock Integration Tests
- Dry-run
- Failure/Timeout Cases
- ตรวจ Regression ของ Macro และ Reward

**Phase 5 — Optional Performance Tuning**

ยังไม่ต้องเริ่ม จนกว่าจะมี Benchmark จาก HP จริง

ให้ทำเป็นรายการข้อเสนอพร้อมต้นทุน/ประโยชน์ก่อน Implement

---

# 11. Expected Deliverables

หลังแก้ไข ให้ส่ง:

1. สรุป Source Audit: What Exists / What's Broken / What's Missing
2. Final State Transition Diagram
3. รายชื่อไฟล์ที่เปลี่ยนและเหตุผล
4. Summary ของ Retry / Fallback Behavior
5. Timing Config Reference พร้อม Default ทุกค่า
6. Logging Schema และตัวอย่าง Failure Log
7. Tests ที่ผ่าน / ไม่ผ่าน / ยังไม่ได้ทดสอบ
8. Performance Metrics ที่วัดได้จริง
9. Remaining Risks และ Optional Improvements

## Final Acceptance Criteria

- Room List และ Create ไม่ FAIL ทันทีเพียงเพราะ Source Template ไม่พบชั่วคราว
- Retry อยู่ภายใต้ Budget/Timeout และไม่คลิกผิด State
- Happy Path ใช้ Position-first ในจุดที่เหมาะสม
- Loading ที่ตรวจไม่ทันไม่ทำให้ FAIL หากมีหลักฐานยืนยัน Destination เพียงพอ
- ไม่มี False Positive จาก HUD READY ของหน้าผิด
- ไม่มี Screenshot/Recognition เพิ่มเพื่อเก็บ Performance Log
- Timing Config ปรับได้และรักษาค่าเดิม
- Macro และ Reward Behavior ไม่เปลี่ยน
- CPU/RAM ไม่ถูกเพิ่มภาระโดยไม่มีเหตุผล
- หาก FAIL ต้องมีข้อมูล Debug ที่ตรวจสอบย้อนหลังได้

**ให้ความสำคัญกับ Reliability และ CPU/RAM มากกว่าการทำ Workflow ให้เร็วขึ้นไม่กี่วินาที**

หากพบว่า Requirement ใดขัดแย้งกัน ให้ยึดข้อจำกัดด้านความปลอดภัยของ State และ Resource Performance ก่อน พร้อมอธิบายสิ่งที่ยังต้องพิสูจน์

อย่าอ้างว่า Optimize สำเร็จหากยังไม่มี Benchmark จริงรองรับ