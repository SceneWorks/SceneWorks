// DOM-backed subset of Locator for exercising the production terminal driver on
// actual React screens. Queries are live; actions use real native DOM setters/events.
import { act } from 'react';
import { click, setSelect } from './dom.js';
const text=node=>(node?.textContent??'').replace(/\s+/g,' ').trim();
const matches=(value,wanted,exact=true)=>wanted instanceof RegExp?wanted.test(value):exact?value===wanted:value.includes(wanted);
const named=(node)=>node.getAttribute('aria-label')??(node.hasAttribute('aria-labelledby')?node.getAttribute('aria-labelledby').split(/\s+/).map(id=>text(node.ownerDocument.getElementById(id))).join(' '):text(node));
const role=node=>node.getAttribute('role')??({BUTTON:'button',SELECT:'combobox',FIELDSET:'group',INPUT:node.type==='checkbox'?'checkbox':undefined}[node.tagName]);
export function terminalBrowserDom(root) {
  const wrap=(query,description='root')=>{
    const one=()=>{const nodes=query();if(nodes.length!==1)throw Error(`${description}: expected one actual DOM control, got ${nodes.length}`);return nodes[0];};
    const enabled=()=>{const node=one();if(node.matches(':disabled')||node.readOnly)throw Error(`${description}: actual DOM control is not editable`);return node;};
    const api={
      locator:selector=>wrap(()=>query().flatMap(node=>[...node.querySelectorAll(selector)]),`${description} ${selector}`),
      filter:({hasText,has}={})=>wrap(()=>query().filter(node=>(!hasText||matches(text(node),hasText,false))&&(!has||has.nodes().some(child=>node.contains(child)))),`${description} filtered`),
      first:()=>wrap(()=>query().slice(0,1),`${description} first`),nodes:query,
      getByRole:(wanted,{name,exact=false}={})=>wrap(()=>query().flatMap(node=>[...node.querySelectorAll('*')]).filter(node=>role(node)===wanted&&(name===undefined||matches(wanted==='group'?text(node.querySelector('legend')):named(node),name,exact))),`role ${wanted} ${name}`),
      getByText:(wanted,{exact=false}={})=>wrap(()=>query().flatMap(node=>[...node.querySelectorAll('*')]).filter(node=>matches(text(node),wanted,exact)&&![...node.children].some(child=>matches(text(child),wanted,exact))),`text ${wanted}`),
      getByLabel:(wanted,{exact=false}={})=>wrap(()=>[...new Set(query().flatMap(node=>[...node.querySelectorAll('label')].filter(label=>matches(text(label),wanted,exact)).map(label=>label.control??label.querySelector('input,select,textarea')).concat([...node.querySelectorAll('[aria-label]')].filter(control=>matches(control.getAttribute('aria-label'),wanted,exact)))).filter(Boolean))],`label ${wanted}`),
      count:async()=>query().length,inputValue:async()=>one().value,isChecked:async()=>one().checked,isEnabled:async()=>!one().matches(':disabled'),isDisabled:async()=>one().matches(':disabled'),getAttribute:async key=>one().getAttribute(key),innerText:async()=>text(one()),allTextContents:async()=>query().map(text),
      waitFor:async({state}={})=>{if(state==='hidden'){if(query().length)throw Error(`${description} still present`);}else one();},
      click:async()=>click(enabled()),selectOption:async value=>act(async()=>{const node=enabled();if(![...node.options].some(option=>option.value===value))throw Error(`${description}: missing exact option ${value}`);setSelect(node,value);}),
      fill:async value=>act(async()=>{const node=enabled(),prototype=node.tagName==='TEXTAREA'?window.HTMLTextAreaElement.prototype:window.HTMLInputElement.prototype;Object.getOwnPropertyDescriptor(prototype,'value').set.call(node,value);node.dispatchEvent(new window.Event('input',{bubbles:true}));}),
      setChecked:async value=>{if(one().checked!==value)await click(enabled());},
    };return api;
  };return wrap(()=>[root]);
}
