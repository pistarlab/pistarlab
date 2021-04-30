<template>
<div>
    <b-card no-body header-bg-variant="info" header-text-variant="white">
        <template v-slot:header>
            <div class="custom-card-header mb-2">{{ agent.ident }}</div>
        </template>
        <b-card-text class="h-100 mt-4 mb-4">
            <b-row>
                <b-col>
                    <div>
                        <b-card-img :src="`/img/agent_cons/${getImageId(agent.ident)}.SVG`" alt="Image" class="svgagent" style="max-height:200px;"></b-card-img>
                    </div>
                    <div class="ml-4 mt-2">

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

                            Notes: <b-link variant="white" size="sm" v-b-modal:edit-notes @click="loadNotes()"><i class="fa fa-edit"></i></b-link> <p>{{agent.notes}}</p>
                        </div>
                        <div class="mt-2">
                            <b-button v-if="agent && !agent.archived" variant="warning" pill @click="updateArchive(true)" size="sm"><i class="fa fa-eye"></i> Move to Archive</b-button>
                            <b-button v-if="agent && agent.archived" variant="secondary" pill @click="updateArchive(false)" size="sm"><i class="fa fa-eye"></i> Unarchive</b-button>
                        </div>
                    </div>
                </b-col>
                <b-col>

                    <div>
                        <div class="data_label mt-2">Spec Id: </div>
                        <b-link target="_blank" :to="`/agent_spec/${agent.specId}`">{{ agent.specId }}</b-link>
                        <div class="data_label mt-2">Seed: </div>
                        {{agent.seed}}
                        <div class="data_label mt-2">Created: </div>{{ agent.created }}
                        <div class="data_label mt-2">Last Checkpoint Id: </div>{{ lastCheckpoint }}

                    </div>

                </b-col>
                <b-col>
                    <h3>Interfaces</h3>
                    <div v-for="(iface, id) in config.interfaces" v-bind:key="id">

                        <div>
                            Id: {{id}}, Type: {{iface.interface_type}}
                        </div>
                        <div class="ml-2">

                            <div class="data_label mt-1">Observation Space: </div>
                            <div class="ml-1" v-if="iface && iface.observation_space">
                                {{iface.observation_space.class_name}} (args={{iface.observation_space.args}},{{iface.observation_space.kwargs}})
                            </div>
                            <div v-else>Undefined</div>

                            <div class="data_label mt-1">Action Space: </div>
                            <div class="ml-1" v-if="iface && iface.action_space">
                                {{iface.action_space.class_name}} (args={{iface.action_space.args}},{{iface.action_space.kargs}})
                            </div>
                            <div v-else>Undefined</div>

                        </div>
                    </div>

                </b-col>
                <b-col>
                    <h3>Components</h3>

                    <div v-for="(component, id) in components" v-bind:key="id">
                        - {{component}}
                    </div>
                    <div v-if="components.length ==0">
                        No components used
                    </div>
                </b-col>
            </b-row>
        </b-card-text>

    </b-card>
</div>
</template>

<script>
import axios from "axios";
import {
    appConfig
} from "../app.config";
import gql from "graphql-tag";
export default {
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
            notes: ""

        };
    },
    mounted() {
        //
    },
    methods: {
        //
        updateArchive(archive) {
            // We save the user input in case of an error
            // const newTag = this.newTag
            // // We clear it early to give the UI a snappy feel
            // this.newTag = ''
            // Call to the graphql mutation
            this.$apollo.mutate({
                // Query
                mutation: gql `mutation archiveMutation($id:String!,$archive:Boolean!) 
                {
                    agentSetArchive(id:$id, archive:$archive){
                        success
                        }
                }`,
                // Parameters
                variables: {
                    id: this.agent.id,
                    archive: archive
                },

            }).then((data) => {
                this.$emit('update')
            }).catch((error) => {
                // Error
                console.error(error)
                // We restore the initial user input
            })
        },
        loadNotes() {
            this.notes = this.agent.notes
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

        getImageId(uid) {
            let id = parseInt(uid.split("-")[1]);
            return id % 19;
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
