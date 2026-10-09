// Mode-2 touch sticks: left altitude/yaw, right level forward/strafe.
export class FlightControls {
  constructor(element, canControl) {
    this.element = element;
    this.canControl = canControl;
    this.media = matchMedia('(pointer: coarse) and (orientation: landscape) and (max-height: 600px)');
    this.controller = new AbortController();
    const options = {signal:this.controller.signal};
    this.sticks = {};
    for (const side of ['left','right']) {
      const zone = element.querySelector(`[data-stick="${side}"]`);
      const state = this.sticks[side] = {zone,knob:zone.querySelector('.stick-knob'),id:null,x:0,y:0};
      const update = e => {
        const rect = zone.getBoundingClientRect(), radius = rect.width * .34;
        let x = (e.clientX-rect.left-rect.width/2)/radius;
        let y = (rect.top+rect.height/2-e.clientY)/radius;
        const length = Math.hypot(x,y);
        if (length > 1) { x /= length; y /= length; }
        state.knob.style.transform = `translate(${x*radius}px,${-y*radius}px)`;
        const magnitude = Math.hypot(x,y), amount = Math.max(0,(magnitude-.12)/.88);
        state.x = magnitude ? x/magnitude*amount : 0;
        state.y = magnitude ? y/magnitude*amount : 0;
      };
      zone.addEventListener('pointerdown', e => {
        if (!this.available() || state.id !== null) return;
        e.preventDefault(); e.stopPropagation();
        state.id=e.pointerId; zone.setPointerCapture(e.pointerId);
        zone.classList.add('active'); update(e);
      },options);
      zone.addEventListener('pointermove', e => {
        if (state.id !== e.pointerId) return;
        e.preventDefault(); e.stopPropagation();
        if (!this.available()) { this.reset(); return; }
        update(e);
      },options);
      for (const name of ['pointerup','pointercancel','lostpointercapture']) {
        zone.addEventListener(name,e => {
          if (state.id === e.pointerId) { e.preventDefault(); e.stopPropagation(); this.release(state); }
        },options);
      }
      zone.addEventListener('contextmenu',e=>e.preventDefault(),options);
    }
    this.media.addEventListener('change',()=>this.reset(),options);
    window.addEventListener('blur',()=>this.reset(),options);
    window.addEventListener('resize',()=>this.reset(),options);
    document.addEventListener('visibilitychange',()=>this.reset(),options);
  }
  available() {
    return this.media.matches && !document.hidden && this.canControl();
  }
  release(state) {
    const id=state.id;
    state.id=null; state.x=0; state.y=0;
    state.zone.classList.remove('active'); state.knob.style.transform='translate(0px,0px)';
    if(id !== null && state.zone.hasPointerCapture(id)) state.zone.releasePointerCapture(id);
  }
  reset() { for (const state of Object.values(this.sticks)) this.release(state); }
  sample() {
    if (!this.available()) this.reset();
    const {left,right}=this.sticks;
    return {yaw:left.x,altitude:left.y,strafe:right.x,forward:right.y};
  }
  dispose() { this.reset(); this.controller.abort(); }
}
