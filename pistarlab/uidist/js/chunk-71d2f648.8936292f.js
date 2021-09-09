(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-71d2f648"],{"2a25":function(t,a,e){"use strict";var s=e("7a0b"),n=e.n(s);n.a},7369:function(t,a,e){"use strict";var s=e("d643"),n=e.n(s);n.a},"7a0b":function(t,a,e){},b036:function(t,a,e){"use strict";e.r(a);var s=function(){var t=this,a=t.$createElement,e=t._self._c||a;return e("div",[e("b-modal",{attrs:{id:"agentnew",size:"xl","hide-header":!0,"hide-footer":!0}},[e("AgentNew",{attrs:{specId:t.agentSpec.ident},on:{agentCreated:function(a){return t.agentCreated(a)}}})],1),e("h1",[e("i",{staticClass:"fa fa-robot"}),t._v(" Agent Spec: "),t.agentSpec?e("span",[t._v(t._s(t.agentSpec.displayedName))]):t._e()]),e("div",{staticClass:"mt-4"}),e("b-button",{directives:[{name:"b-modal",rawName:"v-b-modal:agentnew",arg:"agentnew"}],attrs:{size:"sm",disabled:t.agentSpec.disabled,variant:"primary"}},[t._v("New Instance")]),e("div",{staticClass:"mt-2"}),e("b-container",{attrs:{fluid:""}},[e("b-row",[e("b-col",{staticClass:"text-center",attrs:{cols:"3"}},[e("b-card-img",{staticClass:"mt-4",staticStyle:{"max-width":"300px"},attrs:{src:"/img/agent_spec_icons/agent_"+t.getImageId(t.agentSpec.ident)+".png",alt:"Image"}})],1),e("b-col",[e("div",{staticClass:"pt-2"},[e("div",{staticClass:"data_label"},[t._v("Name")]),e("span",[t._v(t._s(t.agentSpec.displayedName))])]),e("div",{staticClass:"pt-2"},[e("div",{staticClass:"data_label"},[t._v("Spec Id")]),e("span",[t._v(t._s(t.agentSpec.ident))])]),e("div",{staticClass:"pt-2"},[e("div",{staticClass:"data_label"},[t._v("Extension ID")]),e("span",[t._v(t._s(t.agentSpec.extensionId))])]),e("div",{staticClass:"pt-2"},[e("div",{staticClass:"data_label"},[t._v("Version")]),e("span",[t._v(t._s(t.agentSpec.version))])]),e("div",{staticClass:"pt-2"},[e("div",{staticClass:"data_label"},[t._v("Description")]),e("span",{staticStyle:{"white-space":"pre-wrap"}},[t._v(t._s(t.agentSpec.description))])])])],1),e("div",{staticClass:"mt-4"})],1),e("hr"),e("b-button",{directives:[{name:"b-toggle",rawName:"v-b-toggle.collapse-details",modifiers:{"collapse-details":!0}}],attrs:{variant:"secondary"}},[t._v("Configuration Details")]),e("b-collapse",{staticClass:"mt-2",attrs:{id:"collapse-details"}},[e("b-card",[e("b-container",{attrs:{fluid:""}},[e("b-row",[e("b-col",[e("div",{staticClass:"data_label"},[t._v("Default Config")]),t.agentSpec&&t.agentSpec.config?e("div",[e("pre",[t._v(t._s(JSON.parse(t.agentSpec.config)))])]):t._e()])],1),e("div",{staticClass:"mt-4"}),e("b-row",[e("b-col",[e("div",{staticClass:"data_label"},[t._v("Params")]),t.agentSpec&&t.agentSpec.params?e("div",[e("pre",[t._v(t._s(JSON.parse(t.agentSpec.params)))])]):t._e()])],1)],1)],1)],1)],1)},n=[],i=e("8785"),o=e("9530"),c=e.n(o),r=e("c5bd");function l(){var t=Object(i["a"])(["\n  query GetAgentSpec($ident: String!) {\n    agentSpec(ident: $ident) {\n      id\n      ident\n      displayedName\n      description\n      extensionId\n      version\n      config\n      params\n      disabled\n    }\n  }\n"]);return l=function(){return t},t}var d=c()(l()),p={name:"AgentSpecView",components:{AgentNew:r["a"]},apollo:{agentSpec:{query:d,variables:function(){return{ident:this.specId}}}},data:function(){return{agentSpec:{},options:{},config:"",code:"",submitting:!1}},props:{specId:String},methods:{agentCreated:function(t){this.$router.push({path:"/agent/view/".concat(t)})}},watch:{},created:function(){}},v=p,u=(e("2a25"),e("2877")),g=Object(u["a"])(v,s,n,!1,null,"be9c3726",null);a["default"]=g.exports},c5bd:function(t,a,e){"use strict";var s=function(){var t=this,a=t.$createElement,e=t._self._c||a;return t.agentSpec?e("div",[e("b-modal",{attrs:{id:"selectsnapshot",size:"xl",scrollable:"","hide-header":!0,"hide-footer":!1}},[e("div",{staticClass:"text-center mt-4"},[e("div",{staticClass:"mt-4"}),t.errorMessage?e("b-alert",{attrs:{show:"",variant:"danger"}},[t._v(t._s(t.errorMessage)+": "),e("pre",{staticStyle:{"background-color":"inherit"}},[t._v(t._s(t.traceback))])]):t._e(),t.snapshotId?e("span",{staticClass:"h4 mt-4"},[e("i",{staticClass:"fa fa-camera"}),t._v(" Selected Snapshot: "),e("b",[t._v(t._s(t.snapshotId))])]):e("span",{staticClass:"h4 mt-4"},[t._v(" No snapshot selected. A new agent will be created using the default parameters. ")])],1),e("div",{staticClass:"mt-4 ml-4 mb-4"},[e("b-form-radio",{attrs:{value:null,name:"xxx"},model:{value:t.snapshotId,callback:function(a){t.snapshotId=a},expression:"snapshotId"}},[t._v("None")])],1),e("b-tabs",[e("b-tab",{attrs:{title:"Local Snapshots",active:""}},[e("div",{staticClass:"mt-4 overflow-auto"},[e("SnapshotSelector",{attrs:{specId:t.agentSpec.ident},model:{value:t.snapshotId,callback:function(a){t.snapshotId=a},expression:"snapshotId"}})],1)]),e("b-tab",{attrs:{title:"Community Hub Snapshots"}},[e("div",{staticClass:"mt-4 overflow-auto"},[e("SnapshotSelector",{attrs:{specId:t.agentSpec.ident,online:!0},model:{value:t.snapshotId,callback:function(a){t.snapshotId=a},expression:"snapshotId"}})],1)])],1)],1),e("h3",[t._v("New Agent Instance")]),e("div",{staticClass:"mt-4"}),t.$apollo.queries.agentSpec.loading?t._e():e("b-container",{attrs:{fluid:""}},[e("b-row",[e("b-col",{staticClass:"text-center",attrs:{cols:"4"}},[e("b-card-img",{staticStyle:{"max-width":"200px"},attrs:{src:"/img/agent_spec_icons/agent_"+t.getImageId(t.agentSpec.ident)+".png",alt:"Image"}})],1),e("b-col",{staticClass:"text-right",attrs:{cols:"2"}},[e("div",[t._v("Agent Spec:")]),e("div",[t._v("Extension ID:")]),e("div",[t._v("Version:")])]),e("b-col",[e("div",[e("span",[e("router-link",{attrs:{to:"/agent_spec/"+t.agentSpec.ident}},[t._v(t._s(t.agentSpec.displayedName))])],1)]),e("div",[e("span",[t._v(t._s(t.agentSpec.extensionId))])]),e("div",[e("span",[t._v(t._s(t.agentSpec.version))])])])],1),e("div",{staticClass:"mt-2"}),e("b-row",[e("b-col",{staticClass:"text-right",attrs:{cols:"4"}}),e("b-col",[e("div",[e("span",[t._v(t._s(t.agentSpec.description))])])])],1),e("div",{staticClass:"mt-4"}),e("b-row",[e("b-col",[e("div",[e("b-tabs",{attrs:{"content-class":"mt-3",justified:""}},[e("b-tab",{staticClass:"text-center",attrs:{title:"Create",active:""}},[e("p",[t._v("Create an agent with default parameters or select one from an existing snapshot.")]),e("b-row",{staticClass:"mt-4 ml-4 mb-4 "},[e("b-col",[e("b-button",{directives:[{name:"b-modal",rawName:"v-b-modal:selectsnapshot",arg:"selectsnapshot"}],staticClass:"mr-4",attrs:{variant:"info",size:"sm"}},[t._v("Select an Agent Snapshot")])],1),e("b-col",[e("b-form-radio",{staticClass:"mt-2",attrs:{value:null,name:"xxx"},model:{value:t.snapshotId,callback:function(a){t.snapshotId=a},expression:"snapshotId"}},[t._v("New Agent with default parameters.")])],1)],1),e("div",{staticClass:"text-center mt-4"},[e("div",{staticClass:"mt-4"}),t.errorMessage?e("b-alert",{attrs:{show:"",variant:"danger"}},[t._v(t._s(t.errorMessage)+" "),t.traceback?e("pre",{staticStyle:{"background-color":"inherit"}},[t._v(t._s(t.traceback))]):t._e()]):t._e(),t.snapshotId?e("span",{staticClass:"h4 mt-4"},[e("i",{staticClass:"fa fa-camera"}),t._v(" Selected Snapshot"),e("br"),t._v(" "),e("b",[t._v(t._s(t.snapshotId))])]):e("span",{staticClass:"h4 mt-4"},[t._v(" No snapshot selected. A new agent will be created using the default parameters. ")])],1)],1),e("b-tab",{attrs:{title:"Create with Agent Builder"}},[t.params?e("div",[e("div",{staticClass:"ml-3"},[e("ParamEditor",{ref:"peditor",attrs:{params:t.params,values:t.initParamValues},on:{update:function(a){return t.saveParams(a)}}})],1)]):e("div",[t._v(" Agent Builder not supported by this Agent Spec ")]),e("div",{staticClass:"mt-4"}),t.errorMessage?e("b-alert",{attrs:{show:"",variant:"danger"}},[t._v(t._s(t.errorMessage)+": "),e("pre",{staticStyle:{"background-color":"inherit"}},[t._v(t._s(t.traceback))])]):t._e()],1),e("b-tab",{attrs:{title:"Created from raw config"}},[t.config?e("editor",{attrs:{lang:"json",width:"100%",theme:"chrome",height:"600"},on:{init:t.editorInit},model:{value:t.config,callback:function(a){t.config=a},expression:"config"}}):t._e(),e("div",{staticClass:"mt-4"}),e("div",{staticClass:"mt-4"}),t.errorMessage?e("b-alert",{attrs:{show:"",variant:"danger"}},[t._v(t._s(t.errorMessage)+": "),e("pre",{staticStyle:{"background-color":"inherit"}},[t._v(t._s(t.traceback))])]):t._e()],1)],1)],1)])],1)],1),e("div",{staticClass:"mt-4"}),e("b-button-toolbar",[e("div",{staticClass:"ml-auto mr-3 my-auto"},[t._v(" Agent Name: ")]),e("b-form-input",{staticClass:"mr-3",staticStyle:{width:"300px"},attrs:{placeholder:"Enter name or one will be assigned"},model:{value:t.agentName,callback:function(a){t.agentName=a},expression:"agentName"}}),t.submitting?e("b-button",{attrs:{size:"sm",variant:"primary",disabled:""}},[e("b-spinner",{attrs:{small:"",type:"grow"}}),t._v("Processing... ")],1):e("b-button",{attrs:{size:"sm",variant:"primary"},on:{click:t.submit}},[t._v("Create Instance")])],1)],1):t._e()},n=[],i=(e("498a"),e("5530")),o=e("8785"),c=e("bc3a"),r=e.n(c),l=e("4023"),d=(e("bbd0"),e("9530")),p=e.n(d),v=e("321f"),u=function(){var t=this,a=t.$createElement,e=t._self._c||a;return e("div",[t.error?e("b-alert",{attrs:{variant:"danger"}},[t._v(t._s(t.error))]):t._e(),e("b-form-group",[e("b-container",{attrs:{fluid:""}},[e("b-row",[e("b-col",{attrs:{cols:"1"}}),e("b-col",{staticClass:"h4",attrs:{cols:"4"}},[t._v(" Snapshot Info ")]),e("b-col",[e("div",{staticClass:"h4"},[t._v("Environment Stats: ")]),e("b-row",{staticClass:"h6 mb-0 pb-0"},[e("b-col",{attrs:{cols:"4"}},[t._v(" Environment Spec ID ")]),e("b-col",[t._v(" Steps ")]),e("b-col",[t._v(" Episodes ")]),e("b-col",[t._v(" Reward ")]),e("b-col",[t._v(" Sessions ")])],1)],1)],1),e("hr"),t.loading?e("div",[t._v(" Loading... ")]):null==t.snapshots||0==Object.keys(t.snapshots).length?e("div",[t._v(" No snapshots found ")]):t._e(),t._l(t.snapshots,(function(a,s){return e("div",{key:s},[e("b-row",[e("b-col",{staticClass:"text-center",attrs:{cols:"1"}},[e("b-form-radio",{attrs:{name:"xxx",value:a.snapshot_id},on:{change:function(e){return t.select(a.snapshot_id)}},model:{value:t.selected,callback:function(a){t.selected=a},expression:"selected"}})],1),e("b-col",[e("span",{staticClass:"h4"},[t._v(t._s(a.id)+" - "+t._s(a.seed)+" - "+t._s(a.snapshot_version))])])],1),e("b-row",[e("b-col",{attrs:{cols:"1"}}),e("b-col",{attrs:{cols:"4"}},[t.online?t._e():e("div",[a.published?e("span",[e("i",{staticClass:"fa fa-cloud"}),t._v(" Published")]):e("span",[e("i",{staticClass:"fa fa-hdd"}),t._v(" Not Published")])]),e("div",[e("span",[t._v("Created by: "+t._s(a.submitter_id))])]),e("div",[e("span",[t._v(t._s(a.snapshot_description))])])]),e("b-col",{},[null==a.env_stats||0==Object.keys(a.env_stats).length?e("div",[t._v(" No Stats Found ")]):e("div",t._l(a.env_stats,(function(a,s){return e("div",{key:s},[e("b-row",[e("b-col",{attrs:{cols:"4"}},[t._v(" "+t._s(s)+" ")]),e("b-col",[t._v(" "+t._s(a["step_count"])+" ")]),e("b-col",[t._v(" "+t._s(a["episode_count"])+" ")]),e("b-col",[t._v(" "+t._s(a["best_ep_reward_total"])+" ")]),e("b-col",[t._v(" "+t._s(a["session_count"])+" ")])],1)],1)})),0)])],1),e("hr")],1)}))],2)],1)],1)},g=[],b=(e("99af"),e("9b39"),{props:{specId:String,online:{type:Boolean,default:!1}},components:{},data:function(){return{snapshots:[],error:null,selected:null,loading:!0}},mounted:function(){},methods:{loadLocalData:function(){var t=this;console.log("Loading local snapshots"),this.error=null,this.loading=!0,r.a.get("".concat(l["a"].API_URL,"/api/snapshots/list/").concat(this.specId)).then((function(a){t.snapshots=a.data["items"],t.loading=!1})).catch((function(a){t.error=a,t.loading=!1}))},loadOnlineData:function(){var t=this;console.log("Loading online snapshots"),this.error=null,this.loading=!0,r.a.get("".concat(l["a"].API_URL,"/api/snapshots/public/list/spec_id/").concat(this.specId)).then((function(a){t.snapshots=a.data["items"],t.loading=!1})).catch((function(a){t.error=a,t.loading=!1}))},loadData:function(){1==this.online?this.loadOnlineData():this.loadLocalData()},select:function(t){console.log(t),this.$emit("input",t)}},computed:{},created:function(){this.loadData()},beforeDestroy:function(){}}),h=b,m=e("2877"),_=Object(m["a"])(h,u,g,!1,null,"0b6715a4",null),f=_.exports;function S(){var t=Object(o["a"])(["\n  query GetAgentSpec($ident: String!) {\n    agentSpec(ident: $ident) {\n      id\n      ident\n      displayedName\n      description\n      extensionId\n      version\n      config\n      params\n    }\n  }\n"]);return S=function(){return t},t}var C=p()(S()),w={components:{editor:e("7c9e"),SnapshotSelector:f,ParamEditor:v["a"]},apollo:{agentSpec:{query:C,variables:function(){return{ident:this.specId}}}},data:function(){return{agentSpec:{},options:{},config:"",code:"",paramValues:{},submitting:!1,snapshotId:null,snapshot_version:"0",snapshot_description:"",errorMessage:null,agentName:null}},props:{specId:String},computed:{params:function(){return this.agentSpec?JSON.parse(this.agentSpec.params):null},initParamValues:function(){return this.agentSpec.config?JSON.parse(this.agentSpec.config):null}},methods:{editorInit:function(){e("2099"),e("be9d"),e("818b"),e("0329"),e("95b8"),e("79fb"),e("6a21")},cancel:function(){},onError:function(){},selectSnapshot:function(t){this.$bvModal.hide("modal-selectsnapshot"),this.snapshotId=t},submit:function(){var t=this;null!=this.agentName&&""==this.agentName.trim()&&(this.agentName=null);var a={config:JSON.parse(this.config),specId:this.specId,snapshotId:this.snapshotId,name:this.agentName};r.a.post("".concat(l["a"].API_URL,"/api/new_agent_submit"),a).then((function(a){var e=a.data["item"];"uid"in e?t.$emit("agentCreated",e.uid):(console.log("ERROR in response "+JSON.stringify(e)),t.errorMessage=JSON.stringify(e["error"])),t.traceback=e["traceback"],t.submitting=!1})).catch((function(t){this.errorMessage=t,this.submitting=!1}))},saveParams:function(t){this.paramValues=t,this.config=JSON.stringify(Object(i["a"])(Object(i["a"])({},JSON.parse(this.config)),this.paramValues),null,2)}},watch:{agentSpec:function(t){this.agentSpec=t,this.agentSpec.config&&(this.config=JSON.stringify(JSON.parse(this.agentSpec.config),null,2))}},created:function(){}},I=w,x=(e("7369"),Object(m["a"])(I,s,n,!1,null,"6579e9fe",null));a["a"]=x.exports},d643:function(t,a,e){}}]);
//# sourceMappingURL=chunk-71d2f648.8936292f.js.map