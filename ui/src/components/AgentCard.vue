<template>
<div>

    <b-container fluid>
        <b-row>
            <b-col>
                <div class="text-center mt-4">
                    <b-card-img :src="`/img/agent_spec_icons/agent_${getImageId(agent.specId)}.png`" alt="Image" style="max-width:320px; "></b-card-img>
                </div>
                <div class="ml-4 mt-4">

                    <div>
                        <b-modal id="edit-tags" title="edit tags" :hide-footer="true">
                            <b-input-group>
                                <b-input v-model="newTag" style="width:50px" placeholder="Enter a tag" />
                                <b-button class="ml-2" variant="primary" v-b-modal:edit-tags size="sm" @click="modifyTag('add',newTag)">Add</b-button>
                            </b-input-group>
                            <div class="mt-2">
                                <span v-for="(tag,id) in tagList" v-bind:key="id" variant="tag" class="mr-2">
                                    {{tag}} (<b-link @click="modifyTag('remove',tag)">remove</b-link>)
                                    <span v-if="id != tagList.length-1">,
                                    </span>
                                </span>
                            </div>

                        </b-modal>
                        Tags:
                        <b-badge pill v-for="(tag,id) in tagList" v-bind:key="id" variant="tag" class="mr-1">{{tag}}</b-badge>
                        <b-link variant="white" v-b-modal:edit-tags size="sm"><i class="fa fa-edit"></i></b-link>
                    </div>
                    <div class="mt-2">
                        <b-modal id="edit-notes" title="edit notes" @ok="updateNotes()">
                            <b-input-group>
                                <b-textarea v-model="notes" placeholder="Notes" />
                            </b-input-group>
                        </b-modal>

                        Notes: <b-link variant="white" size="sm" v-b-modal:edit-notes @click="loadNotes()"><i class="fa fa-edit"></i></b-link>
                        <p>{{agent.notes}}</p>
                    </div>

                </div>
            </b-col>
            <b-col>

                <div>
                    <h4>Agent Details</h4>
                    <b-modal id="agentname" size="lg" title="Update Agent Name" @ok="updateName()">
                        <p>
                            Agent names should to be human readable and used to help organize your agents. Note they do not need to be unique.
                        </p>
                        <b-form-input v-model="name" placeholder="Enter an agent name" style="width:250px;"></b-form-input>

                    </b-modal>
                    <div class="data_label mt-2">Name: </div>
                    <b-link variant="white" size="sm" v-b-modal:agentname @click="loadName()">
                        <span v-if="agent.name">{{agent.name}}</span><span v-else>None</span>
                    </b-link>

                    <div class="data_label mt-2">Id: </div>{{ agent.ident }}

                    <div class="data_label mt-2">Spec Id: </div>
                    <b-link target="_blank" :to="`/agent_spec/${agent.specId}`">{{ agent.spec.displayedName }}</b-link>
                    <div class="data_label mt-2">Seed: </div>
                    {{agent.seed}}
                    <div class="data_label mt-2">Created: </div>{{ agent.created }}
                    <div class="data_label mt-2">Last Checkpoint Id: </div>{{ lastCheckpoint }}

                </div>

            </b-col>
            <b-col>
                <span class="h4" id="interface_label">Agent Interfaces</span>

                <b-modal id="helpinfo-modal" title="Help" size="lg" ok-only>
                    <HelpInfo contentId="agent_interface" :fullPage="true">
                    </HelpInfo>
                </b-modal>
                <span class="ml-2 text-right">
                    <b-link v-b-modal="'helpinfo-modal'" style="color:white">
                        <i class="fa fa-question-circle"></i>
                    </b-link>
                </span>

                <!-- <b-popover target="interface_label" triggers="hover" placement="left">
                    <template #title>Interfaces</template>
                    Interfaces define how an agent can interact with an environment. Agents that support multiple interfaces can interact with environments with different obvservation/action spaces. For example: an RL agent that can learn from prior gathered experiences as well as run on the environment directly.
                    <HelpInfo contentId="tasks" :fullPage="true">
                    </HelpInfo>

                </b-popover> -->
                <div class="mt-2">
                    <div v-for="(iface, id) in config.interfaces" v-bind:key="id">
                        <span>
                            <h6>{{id}}</h6>
                        </span>
                        <div class="ml-2">
                            <!-- <span class="data_label mt-1">Type: </span>{{iface.interface_type}} -->
                            <div class="data_label mt-1">Observation Space: </div>
                            <div class="ml-1" v-if="iface && iface.observation_space">
                                <SpaceInfo :space="iface.observation_space"></SpaceInfo>
                            </div>
                            <div v-else>Unassigned</div>

                            <div class="data_label mt-1">Action Space: </div>
                            <div class="ml-1" v-if="iface && iface.action_space">
                                <SpaceInfo :space="iface.action_space"></SpaceInfo>

                            </div>
                            <div v-else>Unassigned</div>

                        </div>
                    </div>
                </div>
            </b-col>
            <b-col>
                <h4>Agent Components</h4>
                <div>

                    <div v-for="(component, id) in components" v-bind:key="id">
                        - {{component}}
                    </div>
                    <div v-if="components.length ==0">
                        No components used
                    </div>
                </div>
            </b-col>
        </b-row>
    </b-container>
</div>
</template>

<script>
import axios from "axios";
import {
    appConfig
} from "../app.config";
import gql from "graphql-tag";
import SpaceInfo from "./SpaceInfo.vue";

export default {
    components: {
        SpaceInfo

    },
    props: {
        agent: Object
    },
    data() {
        return {
            componentFields: [{
                    key: "name",
                    label: "Name",
                }, {
                    key: "specId",
                    label: "Spec Id",
                },
                {
                    key: "spec.category",
                    label: "Type",
                }
            ],
            newTag: "",
            notes: "",
            name: null,

        };
    },
    mounted() {
        //
    },
    methods: {
        //

        loadNotes() {
            this.notes = this.agent.notes
        },
        loadName() {
            this.name = this.agent.name
        },
        updateName() {

            this.$apollo.mutate({
                // Query
                mutation: gql `mutation nameMutation($id:String!,$name:String!) 
                {
                    agentSetName(id:$id, name:$name){
                        success
                        }
                }`,
                // Parameters
                variables: {
                    id: this.agent.id,
                    name: this.name
                },

            }).then((data) => {

                this.$emit('update')
            }).catch((error) => {
                // Error
                console.error(error)
            })
        },

        updateNotes() {

            this.$apollo.mutate({
                // Query
                mutation: gql `mutation noteMutation($id:String!,$notes:String!) 
                {
                    agentSetNotes(id:$id, notes:$notes){
                        success
                        }
                }`,
                // Parameters
                variables: {
                    id: this.agent.id,
                    notes: this.notes
                },

            }).then((data) => {

                this.$emit('update')
            }).catch((error) => {
                // Error
                console.error(error)
            })
        },

        modifyTag(action, tag) {

            axios
                .get(
                    `${appConfig.API_URL}/api/agent/${action}/tag/${this.agent.ident}/${tag}`
                )
                .then((response) => {
                    console.log("Here we are")
                    this.$emit('update')
                })
                .catch((e) => {
                    console.log(e);
                    this.error = e;
                });
        }

    },
    computed: {
        tagList() {
            if (this.agent == null) return []
            return this.agent.tags.edges.map(edge => edge.node.tagId)
        },
        lastCheckpoint() {
            if (this.agent && this.agent.lastCheckpoint)
                return JSON.parse(this.agent.lastCheckpoint).id
            else
                return "NA"
        },
        components() {
            const rows = [];
            if (this.agent.components == null) {
                return rows;
            }
            for (const component of this.agent.components.edges) {
                rows.push(component.node);
            }
            return rows;
        },
        config() {
            return JSON.parse(this.agent.config)
        },
        meta() {
            return JSON.parse(this.agent.meta)
        }

    },
    // Fetches posts when the component is created.
    created() {
        //

    },
    beforeDestroy() {
        //

    }
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->

<style lang="scss" scoped>
.data_label {
    padding: 0px;
    margin: 0px
}
</style>
