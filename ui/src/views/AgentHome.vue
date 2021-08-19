<template>
<div class="page">
    <div class="page-content">
        <h1><i class="fa fa-robot"></i> Agents</h1>

        <div class="mt-4"></div>

        <b-button-toolbar>

            <b-modal id="agentnew" size="xl" :hide-header="true" :hide-footer="true">
                <AgentNew :specId="selectedSpecId" @agentCreated="agentCreated($event)"></AgentNew>
            </b-modal>
            <b-modal id="agentspecs" size="xl" :hide-header="true" :hide-footer="true">
                <AgentSpecs @specSelected="newAgentModal($event)"></AgentSpecs>
            </b-modal>
            <b-button variant="primary" size="sm" v-b-modal:agentspecs>New</b-button>
            <!-- <b-button class="ml-2" size="sm" v-b-toggle="`sess_group`">Toggle Session Details</b-button> -->
        </b-button-toolbar>
        <div class="mt-4"></div>

        <b-container fluid>

            <b-row>
                <b-col class="d-flex justify-content-center">
                    <h3>Recently Active</h3>
                    <br />

                </b-col>

            </b-row>
            <b-row>
                <b-col class="d-flex justify-content-center">

                    <div class="mt-2 text-right mr-4">
                        <b-link :to="`/agent/instances`">(view all)</b-link>
                    </div>

                </b-col>

            </b-row>
            <div v-if="$apollo.queries.recentAgents.loading">Loading..</div>
            <div v-else class="">
                <div v-if="items.length > 0">
                    <b-row>
                        <b-col class="d-flex flex-wrap justify-content-center ">
                            <span v-for="item in items" v-bind:key="item.ident" class="m-3">
                                <b-card no-body header-bg-variant="info" header-text-variant="white" class="h-100 card-shadow card-flyer" style="width: 260px">
                                    <template v-slot:header>
                                        <b-button-toolbar>
                                        <span class="custom-card-header mb-2">
                                            <b-link style="color: white" :to="`/agent/view/${item.ident}`">{{ item.ident }}</b-link>
                                        </span>
                                        <b-link v-b-popover.hover.top="'Assign Task'" class="ml-auto mt-1" size="sm" :to="`/task/new/agenttask/?agentUid=${item.ident}`"><i class="fa fa-plus"></i></b-link>
                                        </b-button-toolbar>

                                    </template>
                                    <b-card-text class="h-100">
                                        <b-container class="mt-2">
                                            <b-row>
                                                <b-col class="text-center">
                                                    <b-link :to="`/agent/view/${item.ident}`">
                                                        <b-card-img style="width:80px" :src="`/img/agent_spec_icons/agent_${getImageId(item.specId)}.png`"></b-card-img>
                                                    </b-link>
                                                    <div class="mt-2">
                                                        <b-badge pill v-for="(tag,id) in item.tags.edges" v-bind:key="id" variant="tag" class="mr-1">{{tag.node.tagId}}</b-badge>
                                                        <div v-if="item.tags.edges.length==0" style="height:1.5em"></div>

                                                    </div>
                                                </b-col>
                                            </b-row>
                                            <hr class="mt-1 mb-1" />
                                            <b-row>
                                                <b-col>

                                                    <div class="mt-1 small">

                                                        <span class="data_label mr-1 mt-0">Spec:</span>
                                                        <span class="">
                                                            <b-link :to="`/agent_spec/${item.specId}`">{{ item.spec.displayedName }}</b-link>
                                                        </span>
                                                    </div>

                                                    <div class="mt-1 small">
                                                        <span class="data_label mr-1 mt-0">Created</span>
                                                        <span class="">{{ timedeltafordate(item.created) }} ago</span>
                                                    </div>

                                                    <!-- <b-collapse id="sess_group" class="mt-2 "> -->
                                                    <div class="mt-3 small">
                                                        Recent Sessions
                                                        <div class="ml-2 mt-2 " v-if="item.recentSessions.length ==0">No Sessions Found</div>
                                                        <div v-else class="d-flex flex-wrap ">
                                                            <div class=" mt-2 mr-2 mb-0" v-for="(session, widx) in item.recentSessions" v-bind:key="widx">
                                                                <!-- <span v-if="session.parentSessionId" class="mr-1">
                                                                <b-link v-if="session.parentSessionId" class="" :to="`/session/view/${session.parentSessionId}`" title="parent session"> {{session.parentSessionId}}</b-link>
                                                            </span> -->
                                                                <b-link class="mr-1" :to="`/session/view/${session.ident}`" 
                                                                v-b-popover.hover.top="session.envSpecId">
                                                                    <div style="color:lightgrey" >
                                                                        
                                                                        <span  v-if="session.sessionType != 'RL_SINGLEPLAYER_SESS'"><i class="fas fa-list" title="multiagent"></i> </span>
                                                                        <span v-else><i class="fas fa-cube" title="multiagent"></i> </span>
                                                                        {{session.ident}}
                                                                        <i v-if="session.status=='RUNNING'" class="fa fa-circle ml-1" style="color:lightgreen"></i>
                                                                    </div>
                                                                    <div class="mt-1">

                                                                        <img width=60 
                                                                        :src="`${appConfig.API_URL}/api/env_preview_image/${session.envSpec.ident}`" 
                                                                        alt="" />
                                                                        
                                                                    </div>
                                                                </b-link>

                                                            </div>
                                                        </div>

                                                    </div>
                                                    <!-- </b-collapse> -->

                                                </b-col>
                                            </b-row>

                                        </b-container>
                                    </b-card-text>

                                </b-card>
                            </span>
                        </b-col>
                    </b-row>
                    <b-row>
                        <b-col>

                            <p>{{ error }}</p>
                        </b-col>
                    </b-row>
                </div>
                <div v-else>

                    <b-row>
                        <b-col>
                            No Active Agents Available
                            <div class="mt-4">
                            </div>
                        </b-col>
                    </b-row>
                </div>

            </div>

        </b-container>

    </div>
    <HelpInfo contentId="agents">
    </HelpInfo>

</div>
</template>

<script>
// @ is an alias to /src
import axios from "axios";
import gql from "graphql-tag";
import {
    appConfig
} from "../app.config"

import {
    timedeltafordate,
    timepretty,
    timelength,
    timedelta
} from "../funcs";

import {
    GET_RECENT_AGENTS
} from "../queries";

import AgentSpecs from "../components/AgentSpecs.vue";
import AgentNew from "../components/AgentNew.vue";

export default {
    name: "Agents",
    components: {
        AgentSpecs,
        AgentNew

    },
    apollo: {
        recentAgents: {
            query: GET_RECENT_AGENTS,
            pollInterval: 2000
        },
    },
    data() {
        return {
            recentAgents: [],
            selectedSpecId: null,
            searchQuery: "",
            itemsPerRow: 3,
            n: 13,
            error: "",
            taskEntities: [],
        };
    },
    computed: {
        items() {
            const agents = [];
            this.recentAgents.forEach((agentNode) => {
                const agent = {
                    ...agentNode
                };
                agents.push(agent);
            });
            return agents;
        }
    },
    methods: {

        timedeltafordate,
        timedelta,
        timelength,
        agentCreated(agentId) {
            this.$router.push({
                path: `/agent/view/${agentId}`,
            });

        },
        newAgentModal(specId) {
            this.selectedSpecId = specId
            this.$bvModal.hide("agentspecs")
            this.$bvModal.show("agentnew")
        },

        shutdownAgent(uid) {
            axios
                .get(`${appConfig.API_URL}/api/agent_control/SHUTDOWN?uid=${uid}`)
                .then((response) => {
                    // JSON responses are automatically parsed.
                    this.message = response.data["message"];
                    this.$apollo.queries.recentAgents.refetch();
                })
                .catch((e) => {
                    this.error = e;
                    this.message = this.error;
                });
            console.log(uid);
        }
        // getImageId(uid) {
        //     let id = parseInt(uid.split("-")[1]);
        //     return id % 19;
        // },
    },

    created() {
        // this.$store.dispatch("fetchAgents");
    },
};
</script>
<style>
</style>