<template>
<div>
    <h1>New Agent Task</h1>

    <b-modal id="modal-launch-task" title="Launching Task" size="lg" :hide-footer="true" :hide-header-close="true" :no-close-on-backdrop="true" :no-close-on-esc="true">
        <TaskLoad :uid="newTaskId"></TaskLoad>
    </b-modal>

    <b-modal id="modal-select-envspec" title="Select Environment" size="lg">
        <EnvSelector @click="selectEnvSpec($event)"></EnvSelector>

        <template v-slot:modal-footer="{ cancel }">
            <b-button variant="secondary" @click="cancel()">Cancel</b-button>
        </template>
    </b-modal>

    <b-modal id="modal-select-agent" title="Select Agent" size="lg">
        <AgentSelector @click="loadAgent($event)"></AgentSelector>

        <template v-slot:modal-footer="{ cancel }">
            <b-button variant="secondary" @click="cancel()">Cancel</b-button>
        </template>
    </b-modal>

    <b-modal id="modal-configure-envspec" title="Environment Kwargs" size="lg">
        <b-container fluid>
            <b-row>
                <b-col>
                    <div>
                        
                        These are aguments passed directly to the environment. Changing these may impact behavior and incomparable and inconsistant results.
                        <br/>
                        <br/>
                        JSON format required.
                        <b-textarea v-model="envSpecKwargOverrides" class="mt-4" rows="10" no-auto-shrink size="lg" height="100%"></b-textarea>
                    </div>
                </b-col>
            </b-row>
        </b-container>
    </b-modal>

    <b-modal id="modal-full-config" title="Config Output" size="lg">
        <div>
            <pre class="mt-3" >{{fullConfig}}</pre>
        </div>        
    </b-modal>

    <b-modal id="modal-configure-agent" title="Configure Agent" size="lg">
        <b-tabs content-class="mt-4" justified>
            <b-tab title="Session Configuration">
                <div class="d-flex flex-wrap">
                    <b-form-group style="width:320px" class="ml-2" v-for="item in agentSessionConfigFields" :key="item.key" :label="item.label" :description="item.description">
                        <b-form-checkbox v-if="item.type == 'boolean'" v-model="agentSessionConfig[item.key]" value="true" unchecked-value="false">
                        </b-form-checkbox>
                        <b-form-input v-else v-model="agentSessionConfig[item.key]" :type="item.type" size="sm" class="mt-1"></b-form-input>
                    </b-form-group>
                </div>
            </b-tab>
            <b-tab title="Agent Configuration">
                <ParamEditor interfaceFilter="run" buttonText="Save" :params="selectedAgentParams" :values="agentRunConfig" @update="agentRunConfig = $event"></ParamEditor>
            </b-tab>

            <b-tab title="Environment Wrappers (BROKEN)">

                <strong>WARNING wrappers currently not working. This wrapper need to be updated to support multi agent env interfaces</strong>

                <p>Please select wrappers in the order they will be added the the environment</p>
                <b-form-checkbox-group v-model="agentWrappers" :options="wrapperOptions" name="wrappers" stacked></b-form-checkbox-group>
                <div v-for="(item, i) in agentWrappers" :key="item">
                    <div class="">
                        {{ i }} - {{ wrappers[item]["entry_point"] }}
                    </div>
                </div>

            </b-tab>

        </b-tabs>

        <template v-slot:modal-footer="{  }">
            <b-button variant="primary" @click="updateAgentConfig();">Ok</b-button>
            <b-button variant="danger" @click="removeAgent();">Remove Agent</b-button>

        </template>
    </b-modal>

    <b-modal id="assign-player-modal" title="Assign Player">
        <b-container fluid>
            <b-row>
                <b-col>
                    <div v-if="currentPlayerId != null">
                        {{envPlayers[currentPlayerId].id}}
                        <div v-for="(agent,idx) in agents" v-bind:key="idx" class="m-2">
                            <b-button v-on:click="assignPlayerToAgent(currentPlayerId,idx)">{{agent.ident}}</b-button>
                        </div>
                    </div>
                </b-col>
            </b-row>
        </b-container>
        <template v-slot:modal-footer="{ cancel }">
            <b-button variant="secondary" @click="cancel()">Cancel</b-button>
        </template>
    </b-modal>
    <b-modal id="modal-config" title="Run Configuration" size="lg">
        <div class="d-flex flex-wrap mb-4">
            <b-form-group style="width:300px" class="ml-2" v-for="item in rlSessionConfigFields" :key="item.key" :label="item.label" :description="item.description">
                <b-form-checkbox v-if="item.type == 'boolean'" v-model="sessionConfig[item.key]" value="true" unchecked-value="false">
                </b-form-checkbox>
                <b-form-input v-else v-model="sessionConfig[item.key]" :type="item.type" size="sm" class="mt-1"></b-form-input>
            </b-form-group>
        </div>
        <template v-slot:modal-footer="{ ok }">
            <b-button variant="primary" @click="
            ok();
          ">Ok</b-button>
        </template>
    </b-modal>
    <div>
        <b-container fluid>
            <h3><i class="fas fa-gamepad"></i> Environment</h3>
            <div class="mt-4"></div>
            <b-row>
                <b-col>
                    <div v-if="envSpec">
                        <b-card class="card-shadow card-flyer">
                            <b-container fluid>
                                <b-row>
                                    <b-col cols=3 class="text-center  align-self-center">
                                        <b-card-img :src="`${appConfig.API_URL}/api/env_preview_image/${envSpec.environment.ident}`" alt="" style="max-height:200px; width:auto;"></b-card-img>
                                    </b-col>
                                    <b-col>
                                        <b-container>
                                            <b-row>
                                                <b-col>

                                                    <div class="mt-auto">
                                                        <div class="mb-4">
                                                            <h3>{{envSpec.ident}}</h3>
                                                        </div>

                                                        <div class="data_label">Environment: {{ envSpec.environment.ident }}</div>
                                                        <div class="data_label">Type: {{ envSpec.envType }}</div>

                                                        <div class="data_label" v-if="envMeta.num_players">Num Agents:
                                                            {{envMeta.num_players}}
                                                        </div>

                                                        <div class="data_label">

                                                            <b-modal id="show-obvdetails" size="lg" scrollable>

                                                                <span v-if="envMeta && envMeta.observation_spaces">
                                                                    <pre> {{envMeta.observation_spaces}}</pre>

                                                                </span>

                                                                <span v-else>No observation spaces defined</span>

                                                            </b-modal>
                                                            <b-link v-b-modal.show-obvdetails variant="secondary" size="sm">Observation Space Details</b-link>

                                                        </div>

                                                        <div class="data_label">

                                                            <b-modal id="show-actdetails" size="lg" scrollable>

                                                                <span v-if="envMeta && envMeta.action_spaces">
                                                                    <pre>
                                                                    {{envMeta.action_spaces}}
                                                                    </pre>
                                                                </span>
                                                                <span v-else>No action spaces defined</span>

                                                            </b-modal>
                                                            <b-link v-b-modal.show-actdetails variant="secondary" size="sm">Action Space Details</b-link>

                                                        </div>

                                                    </div>
                                                </b-col>
                                                <b-col>

                                                    <b-container class="border-left">

                                                        <div>
                                                        <div style="max-height:250px;" class="overflow-auto">
                                                            <h3>Players</h3>
                                                            <hr />
                                                            <b-row v-for="(slot,idx) in envPlayers" v-bind:key="idx" class="m-0 mb-2">
                                                                <b-col>{{slot.id}}</b-col>
                                                                <b-col>
                                                                    <span v-if="slot.agent != null">
                                                                        {{agents[slot.agent].ident}}
                                                                    </span>
                                                                    <span v-else>
                                                                        Unassigned
                                                                    </span>
                                                                </b-col>

                                                                <b-col>
                                                                    <b-link size="sm" v-on:click="assignPlayerModal(idx)"><i class="fa fa-edit"></i></b-link>
                                                                </b-col>

                                                            </b-row>

                                                        </div>
                                                        <div v-if="envPlayers == null|| envPlayers.length==0">
                                                            <b-row>
                                                                <b-col>
                                                                    No Players Found
                                                                </b-col>
                                                            </b-row>
                                                        </div>
                                                        </div>
                                                        <div v-if="envSpec && envSpec.envType== 'RL_SINGLEPLAYER_ENV'" class="mt-2">
                                                            <hl/>
                                                             <h4>Environment Instances</h4>

                                                            <b-form-input v-model="batchSize" type="number" style="width:100px" class="ml-2 pl-5"></b-form-input>
                                                        </div>
                                                    </b-container>

                                                </b-col>
                                            </b-row>
                                        </b-container>
                                    </b-col>
                                </b-row>
                            </b-container>
                        </b-card>
                        <div class="mt-4"></div>
                        <b-button size="sm" variant="info" v-b-modal.modal-select-envspec>Change Environment</b-button>
                        <b-button size="sm" class="ml-2"  variant="secondary" v-b-modal.modal-configure-envspec>Edit Arguments</b-button>


                    </div>
                    <div v-else>
                        <b-card body-text-variant="" v-b-modal.modal-select-envspec class="h-100 card-shadow card-flyer" style="width: 100px;background-color:#ccc;color:#000">
                            <b-card-body class="text-center">

                                <div>
                                    <i class="fa fa-plus"></i>
                                </div>
                            </b-card-body>

                        </b-card>
                    </div>

                </b-col>
            </b-row>

            <div class="mt-4"></div>

            <hr />
            <div class="mt-4"></div>
            <h3><i class="fas fa-robot"></i> Agents</h3>

            <div class="mt-4"></div>

            <b-row>
                <b-col class="d-flex flex-wrap mb-4">
                    <span v-for="(agent,idx) in agents" v-bind:key="idx" class="mr-3">
                        <b-card no-body header-bg-variant="info" header-text-variant="white" type="button" class="h-100 card-shadow card-flyer stretched-link" style="width: 320px" v-on:click="agentConfigModal(idx)">
                            <template v-slot:header>
                                <div class="custom-card-header  mb-2">
                                    {{agent.ident}}
                                </div>

                            </template>

                            <b-container class="mt-2">
                                <b-row>
                                    <b-col class="text-center">

                                        <b-card-img :src="`/img/agent_spec_icons/agent_${getImageId(agent.specId)}.png`" alt="Image" style="width:100px;">
                                        </b-card-img>
                                    </b-col>

                                </b-row>
                                <div class="mt-3"></div>
                                <b-row>
                                    <b-col>
                                        <div>
                                            <span class="data_label">Spec Id: </span> {{agent.specId}}
                                        </div>
                                        <div class="data_label">

                                        </div>
                                        <div class="data_label">
                                            Players Assigned: {{getPlayersForAgent(idx).join(" ")}}
                                        </div>

                                    </b-col>
                                </b-row>
                            </b-container>

                        </b-card>
                    </span>
                    <span>
                        <b-card body-text-variant="" class="h-100 card-shadow card-flyer  " style="width: 100px;background-color:#ccc;color:#000" v-b-modal.modal-select-agent>
                            <b-card-body class="text-center align-middle">
                                <i class="fa fa-plus"></i>
                            </b-card-body>
                        </b-card>
                    </span>

                </b-col>
            </b-row>

            <div class="mt-4"></div>

            <div v-if="Object.keys(agents).length>1">
                <hr />

                <div class="mt-2 mb-3">
                    <div class="h-5">Multi Agent Session Configuration <b-button size="sm" variant="white" v-b-modal.modal-config><i class="fa fa-edit"></i></b-button>
                    </div>
                </div>
            </div>
            <div class="mt-4"></div>
            <hr />
            <b-row>
                <b-col>

                    <b-alert show variant="danger" v-if="errorMessage">
                        Submission Failed: {{ errorMessage }}
                        <div>
                            <pre class="error">{{ traceback }}</pre>
                        </div>
                    </b-alert>
                    <div class="ml-auto">
                        <b-button size="sm" v-if="!submitting" variant="primary" v-on:click="sendData">Submit</b-button>
                        <b-button v-else variant="primary" disabled>
                            <b-spinner small type="grow"></b-spinner>Processing...
                        </b-button>

                        <b-button size="sm" class="ml-2" variant="secondary" v-on:click="showFullConfig()">View Config Output</b-button>

                    </div>
                </b-col>
            </b-row>
        </b-container>
    </div>
    <div></div>
</div>
</template>

<script lang="ts">
// @ is an alias to /src
import axios from "axios";
import {
    appConfig
} from "../app.config";
import {
    timedelta,
    timepretty
} from "../funcs";
import gql from "graphql-tag";

const rlSessionConfigFields = [{
        key: "max_episodes",
        label: "Max Episodes",
        description: "Max Episodes before Termination",
        defaultValue: null,
        type: "number",
    },
    {
        key: "max_steps",
        label: "Max Steps",
        description: "Max Steps before Termination",
        defaultValue: null,
        type: "number",
    },
    {
        key: "max_steps_in_episode",
        label: "Max Steps per Episode",
        description: "Max Steps per Episode",
        defaultValue: null,
        type: "number",
    },
    {
        key: "episode_record_freq",
        label: "Ep Record Freq",
        description: "Frequency of Episode Recordings",
        defaultValue: 50,
        type: "number",
    },
    {
        key: "episode_record_preview_interval",
        label: "Episode Record Preview Interval",
        description: "Interval for preview generation",
        defaultValue: 1,
        type: "number",
        isNumber: false,
    },
    {
        key: "step_log_freq",
        label: "Step Log Freq",
        description: "Frequency of Step Logging",
        defaultValue: 30,
        type: "number",
    },
    {
        key: "episode_log_freq",
        label: "Episode Log Freq",
        description: "Frequency of Episode Summary Logging",
        defaultValue: 1,
        type: "number",
    },

    {
        key: "preview_rendering",
        label: "Use Preview Rendering",
        description: "Enable Preview Rendering using environment render function",
        defaultValue: true,
        type: "boolean",
    },
    {
        key: "frame_stream_enabled",
        label: "Enable Live Frame Stream",
        description: "Enable Live Frame Stream (preview rendering must also be enabled)",
        defaultValue: true,
        type: "boolean",
    },
];

const dataSessionConfigFields = [{
        key: "max_epochs",
        label: "Max Epochs",
        description: "Max Epochs before Termination",
        defaultValue: null,
        type: "number",
    },
    {
        key: "max_batches",
        label: "Max Batches",
        description: "Max Batches before Termination",
        defaultValue: null,
        type: "number",
    }
];

function getSessionConfigDefaults(sessionConfigDefaults) {
    const defaultValues = {};
    sessionConfigDefaults.forEach((item) => {
        defaultValues[item.key] = item.defaultValue;
    });
    return defaultValues;
}

const GET_WRAPPERS = gql `
  {
    wrappers: envWrappers {
      entry_point
      kwargs
    }
  }
`;

const GET_ENV_SPEC = gql `
  query GetEnvSpec($ident: String!) {
    envSpec(ident: $ident) {
      id
      ident
      displayedName
      config
      meta
      envType
      environment {
          id
        ident
      }
    }
  }
`;

const GET_TASK = gql `
  query GetTask($ident: String!) {
    task(ident: $ident) {
      id
      ident
      specId
      config
    }
  }
`;

const GET_AGENT = gql `
  query GetAgent($ident: String!) {
    agent(ident: $ident) {
        id
          ident
          created
          specId
          config
            spec{
                id
                params
            }
    }
  }
`;

import AgentSelector from "../components/AgentSelector2.vue";
import EnvSelector from "../components/EnvSelector.vue";
import ParamEditor from "../components/ParamEditor.vue";

import TaskLoad from "../components/TaskLoad.vue";

export default {
    name: "AgentTaskNew",
    components: {
        AgentSelector,
        EnvSelector,
        TaskLoad,
        ParamEditor
    },
    apollo: {
        wrappers: GET_WRAPPERS,
        envSpec: {
            query: GET_ENV_SPEC,
            skip: true,
            variables() {
                return {
                    ident: this.selectedEnvSpecId,
                };
            },
        },
    },
    data() {
        return {
            appConfig,
            selectedEnvSpecId: null,
            envSpec: null,
            envSpecKwargOverrides: "{}",
            batchSize: 1,

            agents: {},
            agentConfigs: {},

            currentAgentIdx: null,
            selectedAgent: null,

            agentWrappers: [],
            agentRunConfig: {},
            agentSessionConfig: {},
            agentSessionConfigFields: [],

            wrappers: [],

            currentPlayerId: null,
            envPlayers: [],
            envMeta: {},

            errorMessage: "",
            traceback: "",
            submitting: false,
            newTaskId: null,

            rlSessionConfigFields,
            dataSessionConfigFields,
            sessionConfig: getSessionConfigDefaults(rlSessionConfigFields),

            fullConfig: "{}"
        };
    },
    props: {
        uid: String,
        agentUid: String,
        envSpecId: String,
    },
    computed: {
        wrapperOptions() {
            return this.wrappers.map((el, i) => {
                return {
                    value: i,
                    text: el["entry_point"],
                };
            });
        },

        checkSpaceMatch() {
            if (this.envSpec == null || this.agent == null) {
                return "NA";
            } else {
                const envObsSpace = JSON.parse(this.envSpec.meta).observation_space;
                const agentObsSpace = JSON.parse(this.agent.config).observation_space;

                if (JSON.stringify(envObsSpace) == JSON.stringify(agentObsSpace)) {
                    return "Compatible";
                } else {
                    return "Incompatible";
                }
            }
        },

        selectedAgentParams() {
            if (!this.selectedAgent || !this.selectedAgent.spec) {
                return null
            } else {
                return JSON.parse(this.selectedAgent.spec.params)
            }
        },
        // initParamValues() {
        //     if (!this.selectedAgent || !this.selectedAgent.config || this.selectedAgent.config == '') return {}

        //     return JSON.parse(this.selectedAgent.config)
        // }
    },
    watch: {
        wrappers: function (val) {
            this.wrappers = val;
        },
        envSpec: function (val) {
            this.envSpec = val
            if (this.envSpec) {
                this.envMeta = JSON.parse(this.envSpec.meta)
                this.envSpecKwargOverrides = JSON.stringify(JSON.parse(this.envSpec.config).env_kwargs,null,2)
                this.resetPlayers();
            }
        }
    },
    methods: {

        showFullConfig() {
            this.fullConfig = JSON.stringify(this.prepData(), null, 2)
            this.$bvModal.show("modal-full-config");

        },
        agentConfigModal(idx) {
            this.currentAgentIdx = idx
            let agent = this.agents[this.currentAgentIdx]
            console.log(JSON.stringify(agent, null, 2))
            this.selectedAgent = agent

            this.agentSessionConfigFields = agent['session_config_fields']

            this.agentWrappers = agent['wrappers']
            this.agentRunConfig = agent["run_config"]
            this.agentSessionConfig = agent["session_config"]
            this.$bvModal.show("modal-configure-agent");

        },
        updateAgentConfig() {
            let agent = this.agents[this.currentAgentIdx]

            agent['wrappers'] = this.agentWrappers
            agent["session_config"] = this.agentSessionConfig
            agent["run_config"] = this.agentRunConfig

            agent["config"] = this.selectedAgentConfig
            this.$set(this.agents, agent.ident, agent)
            this.$bvModal.hide("modal-configure-agent");

        },
        selectEnvSpec(uid) {
            this.$bvModal.hide("modal-select-envspec");
            this.selectedEnvSpecId = uid;
            this.$apollo.queries.envSpec.skip = false;
            this.$apollo.queries.envSpec.refetch();
        },

        selectAgent(agent) {
            console.log("Select Agent")
            console.log(JSON.stringify(agent));
            if (Object.keys(this.agents).every((k) => k != agent.ident)) {
                this.$set(this.agents, agent.ident, agent)
                agent['run_config'] = JSON.parse(agent.config)
                console.log(JSON.stringify(agent['run_config'], null, 2))
                agent['interface_type'] = agent['run_config']['interfaces']['run']['interface_type']
                let agentSessionConfigFields = (agent['interface_type'] == "rl") ? this.rlSessionConfigFields : this.dataSessionConfigFields
                agent['session_config'] = getSessionConfigDefaults(agentSessionConfigFields)
                agent['session_config_fields'] = agentSessionConfigFields
                agent['wrappers'] = []
                this.autoAssign()

                this.$bvModal.hide("modal-select-agent");
            }

        },
        assignPlayerModal(playerId) {
            console.log(playerId)
            this.currentPlayerId = playerId
            this.$bvModal.show("assign-player-modal");

        },
        assignPlayerToAgent(playerIdx, agentIdx) {
            this.$set(this.envPlayers[playerIdx], 'agent', agentIdx)
            this.$bvModal.hide("assign-player-modal");
            // Vue.set(this.agents,agentIdx,this.agents[agentIdx])
        },
        getPlayersForAgent(agentIdx) {
            let results = []
            this.envPlayers.forEach((player) => {
                if (player.agent == agentIdx) {
                    results.push(player.id)
                }
            })
            if (results.length == 0)
                results.push("None")
            return results

        },
        resetPlayers() {
            if (this.envSpec && 'possible_players' in this.envMeta) {
                const meta = this.envMeta;
                let results = [];

                meta.possible_players.forEach((id) => {
                    const result = {
                        id: id,
                        observation_space: meta.observation_spaces[id],
                        action_space: meta.action_spaces[id],
                    };
                    results.push(result);
                });
                this.envPlayers = results;
                this.autoAssign()

            } else
                this.envPlayers = [];

        },
        prepData() {
            try {

                let agentConfigList = []
                Object.values(this.agents).forEach((agent:any) => {
                    let agentConfig = {}
                    agentConfig['ident'] = agent.ident
                    agentConfig['run_config'] = agent.run_config
                    agentConfig['session_config'] = agent.session_config
                    agentConfig['session_config']["wrappers"] = agent.wrappers.map(
                        (i) => this.wrappers[i]
                    );

                    //Fix types
                    agent.session_config_fields.forEach((item) => {
                        if (item.type == "number") {
                            agentConfig['session_config'][item.key] = Number(agentConfig['session_config'][item.key]);
                        }
                    });
                    agentConfigList.push(agentConfig)

                })

                //Fix types
                this.rlSessionConfigFields.forEach((item) => {
                    if (item.type == "number") {
                        this.sessionConfig[item.key] = Number(this.sessionConfig[item.key]);
                    }
                });

                let playerAssignments = {}
                this.envPlayers.forEach((player) => {
                    playerAssignments[player.id] = player.agent
                })

                const outgoingData = {
                    agents: agentConfigList,
                    env_spec_id: this.envSpec.ident,
                    env_kwargs: JSON.parse(this.envSpecKwargOverrides),
                    session_config: this.sessionConfig,
                    player_assignments: playerAssignments,
                    batch_size :parseInt(this.batchSize)
                };
                console.log("SENDING " + JSON.stringify(outgoingData));
                return outgoingData;
            } catch (error) {
                return {
                    'error': error
                }
            }

        },
        sendData() {
            this.submitting = true;
            console.log("sending data");
            try {
                const outgoingData = this.prepData();
                axios
                    .post(
                        `${appConfig.API_URL}/api/submit_agent_task`,
                        outgoingData
                    )
                    .then((response) => {
                        const data = response.data["item"];
                        if ("uid" in data) {
                            this.newTaskId = data.uid
                            this.$bvModal.show("modal-launch-task");
                            // this.$router.push({
                            //     path: `/task/load/${data.uid}`,
                            // });
                        } else {
                            console.log("ERROR in response " + JSON.stringify(data));
                            this.errorMessage = JSON.stringify(data["error"]);
                        }
                        this.traceback = data["traceback"];

                        this.submitting = false;
                    })
                    .catch(function (error) {
                        this.errorMessage = error;
                        this.submitting = false;
                    });
            } catch (error) {
                console.log("ERROR " + error);
                this.error = error;
                this.submitting = false;
            }
        },
        loadTaskInfo() {
            if (this.uid == null) {
                return;
            }
            console.log("TODO");

        },
        loadAgent(agentId) {
            console.log("Preloading agent " + agentId)
            this.$apollo.query({
                query: GET_AGENT,
                variables: {
                    ident: agentId
                }
            }).then((response) => {
                console.log(response)
                this.selectAgent(response.data.agent)
            }).catch((response) => {

                this.error = "ERROR GETTING AGENT  " + response;
                console.log(this.error)
            })
        },
        duplicateTask(taskId) {
            console.log("Preloading task " + taskId)
            this.$apollo.query({
                query: GET_TASK,
                variables: {
                    ident: taskId
                }
            }).then((response) => {
                const taskConfig = JSON.parse(response.data.task.config)
                console.log(JSON.stringify(taskConfig, null, 2))
                if (taskConfig.agents)
                    taskConfig.agents.forEach((agentData) => this.loadAgent(agentData.ident))
                else this.loadAgent(taskConfig.agent_id)

                this.selectEnvSpec(taskConfig.env_spec_id)
                this.sessionConfig = taskConfig['session_config']

                this.envSpecKwargOverrides = JSON.stringify(taskConfig['env_kwargs'])
            }).catch((response) => {

                this.error = "ERROR GETTING TASK with id " + taskId;
                console.log(this.error)
            })
        },
        removeAgent() {
            this.$delete(this.agents, this.currentAgentIdx)
            this.$bvModal.hide("modal-configure-agent");
            this.autoAssign()

        },
        autoAssign() {

            if (this.envPlayers.length == 0 || this.agents.length == 0) {
                return
            }
            const agentKeys = Object.keys(this.agents).map((key, idx) => key)

            this.envPlayers.forEach((player, idx) => {
                console.log("p " + idx + " agent: " + agentKeys)
                this.$set(this.envPlayers[idx], 'agent', agentKeys[idx % agentKeys.length]) //agentCount % idx)
            })

        }
        // manualSelectEnvSpec(specId) {
        //     console.log("Preloading " + specId)
        //     this.$apollo.query({
        //         query: GET_ENV_SPEC,
        //         variables: {ident: specId}
        //     }).then((response) => {
        //         this.selectEnvSpec(response.data.envSpec)
        //     }).catch((response) => {

        //         this.error = "ERROR GETTING ENVSPEC";
        //         console.log(this.error)
        //     })
        // }
    },
    created() {
        this.submitting = false;

        console.log("Env Spec Id " + this.envSpecId);
        console.log("Agent UID: " + this.agentUid);
        console.log("Task UID: " + this.uid);

        if (this.uid) {
            this.duplicateTask(this.uid)
        }

        if (this.envSpecId) {
            this.selectEnvSpec(this.envSpecId)

        }
        if (this.agentUid) {
            this.loadAgent(this.agentUid)
        }
    },
};
</script>

<style scoped>
.truncate {
    width: 500px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
</style>
